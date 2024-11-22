import os
import json
import numpy as np
from tqdm import tqdm
import random
import torch.utils.data as data
import numpy as np
from torch_geometric.nn.pool import knn_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.pifold_module_sae import GeoFeaturizer


class MyTokenizer:
    def __init__(self):
        self.alphabet_protein = 'ACDEFGHIKLMNPQRSTVWY' # [X] for unknown token
        self.alphabet_RNA = 'AUGC'
    
    def encode(self, seq, RNA=False):
        def safe_indexing(alphabet, s):
            if s in alphabet:
                return alphabet.index(s)
            else:
                return 20 if not RNA else 4

        if RNA:
            return [safe_indexing(self.alphabet_RNA, s) for s in seq]
        else:
            return [safe_indexing(self.alphabet_protein, s) for s in seq]

class PDBDataset(data.Dataset):
    def __init__(self, data_path='./',  split='train', min_length=30, max_length=500,  test_name='All', data = None, removeTS=0, version=4.2, k_neighbors=30, patch_idx=0, AF2DB=False):
        self.__dict__.update(locals())
        self.tokenizer = MyTokenizer()
        if data is None:
            self.all_data = self.cache_data()
            self.data = self.all_data[split]
        else:
            self.data = data
        
        self.data_num = len(self.data)
        
    def read_line_Af2DB(self, line, alphabet_set, min_length=30, max_length=1024):
        entry = json.loads(line)
        seq = entry['seq']

        chain_encoding = torch.ones(len(seq))

        if (min_length<=len(entry['seq'])) and (len(entry['seq']) <= max_length):
            N = np.array(entry['N'])
            CA = np.array(entry['CA'])
            C = np.array(entry['C'])
            O = np.array(entry['O'])
            seq = torch.tensor(self.tokenizer.encode(entry['seq']))
            X = torch.from_numpy(np.stack([N, CA, C, O], axis=1)).float()
            mask = ~torch.isnan(X.sum(dim=(1,2)))

            return {
                'title':entry['title'],
                'seq':seq[mask],
                'X':X[mask],
                'chain_encoding': chain_encoding[mask]
            }
    
    def read_line_multi_chain(self, line, alphabet_set, min_length=30, max_length=1024):
        entry = json.loads(line)
        seq = entry['seq']

        for key, val in entry['coords'].items():
            entry['coords'][key] = np.asarray(val)

        chain_encoding = []
        for i, L in enumerate(entry['chain_length'].values()):
            chain_encoding.extend([i]*L)
        chain_encoding = torch.tensor(chain_encoding)

        if (min_length<=len(entry['seq'])) and (len(entry['seq']) <= max_length):
            N = entry['coords']['N']
            CA = entry['coords']['CA']
            C = entry['coords']['C']
            O = entry['coords']['O']
            seq = torch.tensor(self.tokenizer.encode(entry['seq']))
            X = torch.from_numpy(np.stack([N, CA, C, O], axis=1)).float()
            mask = ~torch.isnan(X.sum(dim=(1,2)))

            return {
                'title':entry['name'],
                'seq':seq[mask],
                'X':X[mask],
                'chain_encoding': chain_encoding[mask]
            }

    def read_line_single_chain(self, line, alphabet_set, min_length=30, max_length=1024):
        entry = json.loads(line)
        seq = entry['seq']

        for key, val in entry['coords'].items():
            entry['coords'][key] = np.asarray(val)


        chain_list = []
        start = 0
        for i, L in enumerate(entry['chain_length'].values()):
            N = entry['coords']['N'][start: start+L]
            CA = entry['coords']['CA'][start: start+L]
            C = entry['coords']['C'][start: start+L]
            O = entry['coords']['O'][start: start+L]
            seq = torch.tensor(self.tokenizer.encode(entry['seq'][start: start+L]))
            X = torch.from_numpy(np.stack([N, CA, C, O], axis=1)).float()
            if X.shape[0]==0:
                continue
            chain_encoding = torch.from_numpy(np.array([i]*L))
            mask = ~torch.isnan(X.sum(dim=(1,2)))

            data = {
                    'title':entry['name'].split('.')[0]+f'_{i}',
                    'seq':seq[mask],
                    'X':X[mask],
                    'chain_encoding': chain_encoding[mask]
                }
            
            start += L
            if (min_length<=len(data['seq'])) and (len(data['seq']) <= max_length):
                chain_list.append(data)
        return chain_list


    def cache_data(self):

        data_dict = {'train': [], 'val': [], 'test': []}
        alphabet='ACDEFGHIKLMNPQRSTVWY'
        alphabet_set = set([a for a in alphabet])

        if not os.path.exists(self.data_path):
            raise "no such file:{} !!!".format(self.data_path)
        else:
            if self.data_path.endswith('.jsonl'):
                with open(self.data_path) as f:
                    lines = f.readlines()
            else:
                with open(self.data_path+'/pdb.jsonl') as f:
                    lines = f.readlines()

            # data_list = []
            # for idx, line in tqdm(enumerate(lines)):
            #     data = self.read_line_multi_chain(line, alphabet_set)
            #     if (data is not None) and (data['X'].shape[0]>=30):
            #         data_list.append(data)
        print('-------- finish load data ---------')
        data_dict['test'] = lines[:1000]
        data_dict['val'] = lines[:1000]
        data_dict['train'] = lines
        return data_dict
    
    def __len__(self):
        # return len(self.data)
        return self.data_num
    
    def _get_features(self, batch):
        S,  X = batch['seq'], batch['X']

        X, S = X.unsqueeze(0), S.unsqueeze(0)
        mask = torch.isfinite(torch.sum(X,(2,3))).float() # atom mask
        numbers = torch.sum(mask, axis=1).int()
        S_new = torch.zeros_like(S)
        X_new = torch.zeros_like(X)+torch.nan
        for i, n in enumerate(numbers):
            X_new[i,:n,::] = X[i][mask[i]==1]
            S_new[i,:n] = S[i][mask[i]==1]

        X = X_new
        S = S_new
        isnan = torch.isnan(X)
        mask = torch.isfinite(torch.sum(X,(2,3))).float()
        X[isnan] = 0.

        mask_bool = (mask==1)
        def node_mask_select(x):
            shape = x.shape
            x = x.reshape(shape[0], shape[1],-1)
            out = torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])
            out = out.reshape(-1,*shape[2:])
            return out

        batch_id = torch.arange(mask_bool.shape[0], device=mask_bool.device)[:,None].expand_as(mask_bool)

        seq = node_mask_select(S)
        X = node_mask_select(X)
        batch_id = node_mask_select(batch_id)
        mask = torch.masked_select(mask, mask_bool)

        C_a = X[:,1,:]
        edge_idx = knn_graph(C_a, k=self.k_neighbors, batch=batch_id, loop=True, flow='target_to_source')
        batch={
               'title': batch['title'],
                'X':X,
                'S': seq,
                'edge_idx':edge_idx,
                'batch_id':batch_id,
                'mask':mask,
                'num_nodes':torch.tensor(X.shape[0]).reshape(1,),
                'chain_encoding': batch['chain_encoding']}
        return batch
    
    def __getitem__(self, index):
        line = self.data[index]
        alphabet='ACDEFGHIKLMNPQRSTVWY'
        alphabet_set = set([a for a in alphabet])
        try:
            item = self.read_line_multi_chain(line, alphabet_set)
        except:
            item = self.read_line_Af2DB(line, alphabet_set)
        if (item is None) or (item['X'].shape[0]<self.min_length):
            return None

        L = len(item['seq'])
        

        if L>self.max_length:
            # 计算截断的最大索引
            max_index = L - self.max_length
            # 生成随机的截断索引
            truncate_index = random.randint(0, max_index)
            # 进行截断
            item['seq'] = item['seq'][truncate_index:truncate_index+self.max_length]
            item['X'] = item['X'][truncate_index:truncate_index+self.max_length]
            item['chain_encoding'] = item['chain_encoding'][truncate_index:truncate_index+self.max_length]
        


        data = self._get_features(item)
        
        # V, E, T, T_ts, batch_id, edge_idx, chain_encoding = GeoFeaturizer.from_X_to_features(data['X'], data['edge_idx'], data['batch_id'], data['chain_encoding'], virtual_frame_num=3)
        # data['V'] = V
        # data['E'] = E
        # data['T_ts_rots'] = T_ts._rots._rot_mats
        # data['T_ts_trans'] = T_ts._trans
        # data['batch_id_extend'] = batch_id
        # data['edge_idx_extend'] = edge_idx
        # data['chain_encoding'] = chain_encoding
        # print(data['title'])
        return data