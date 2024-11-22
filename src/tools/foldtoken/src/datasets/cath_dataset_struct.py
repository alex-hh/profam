import os
import json
import numpy as np
from tqdm import tqdm
import random
import torch.utils.data as data
import numpy as np
from torch_geometric.nn.pool import knn_graph
import torch
import lmdb
import os.path as osp
from src.chroma.data import Protein
import glob
import pickle
from transformers import AutoTokenizer
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

class CATHDataset(data.Dataset):
    def __init__(self, data_path='./',  split='train', max_length=500, test_name='All', data = None, removeTS=0, version=4.2, k_neighbors=30, mixAFDB=0):
        self.__dict__.update(locals())
        self.db = None
        self.fnames = []
        if data is None:
            self.all_data = self.cache_data(self.data_path)
            self.data = self.all_data[split]
        else:
            self.data = data
        
        self.tokenizer = MyTokenizer()
        
        # self.data = self.data[:1000]
        self.cath_len = len(self.data)
        self.afdb_len = len(self.fnames)
        self.all_len = self.cath_len+self.afdb_len
    
    
    def read_line(self, line, alphabet_set, min_length=30, max_length=1024):
        entry = json.loads(line)

        for key, val in entry['coords'].items():
            entry['coords'][key] = np.asarray(val)


        chain_length = len(entry['seq'])
        chain_encoding = torch.zeros(chain_length)
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


    def cache_data(self, data_path):
        data_dict = {'train': [], 'val': [], 'test': []}
        if not os.path.exists(data_path):
            raise "no such file:{} !!!".format(data_path)
        else:
            with open(data_path+'/chain_set.jsonl') as f:
                lines = f.readlines()
        data_dict['test'] = lines[:1000]
        data_dict['val'] = lines[:1000]
        data_dict['train'] = lines
        return data_dict
    
    def __len__(self):
        return self.all_len
    
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
        batch={ 'title': batch['title'],
                'X':X,
                'S': seq,
                'edge_idx':edge_idx,
                'batch_id':batch_id,
                'mask':mask,
                'num_nodes':torch.tensor(X.shape[0]).reshape(1,),
                'chain_encoding': torch.zeros_like(batch_id)}
        return batch
    
    def sv2pdb(self, item):
        X = np.stack([item['N'], item['CA'], item['C'], item['O']], axis=1)
        X = torch.tensor(X)
        C = torch.ones_like(X[:, 0,0])
        from src.chroma import constants
        S = torch.tensor([constants.AA20.index(one) for one in item['seq']])
        protein = Protein(X[None], C[None], S[None], device='cuda')
        name = item['title']
        protein.to_PDB(f'/huyuqi/xmyu/VQProteinFormer/step1_VQDiff/pdbs/cath4.3/{name}.pdb')

    def load_cath_data(self, index):
        item = self.data[index]
        L = len(item['seq'])
        if L>self.max_length:
            # 计算截断的最大索引
            max_index = L - self.max_length
            # 生成随机的截断索引
            truncate_index = random.randint(0, max_index)
            # 进行截断
            item['seq'] = item['seq'][truncate_index:truncate_index+self.max_length]
            item['CA'] = item['CA'][truncate_index:truncate_index+self.max_length]
            item['C'] = item['C'][truncate_index:truncate_index+self.max_length]
            item['O'] = item['O'][truncate_index:truncate_index+self.max_length]
            item['N'] = item['N'][truncate_index:truncate_index+self.max_length]
            item['chain_mask'] = item['chain_mask'][truncate_index:truncate_index+self.max_length]
            item['chain_encoding'] = item['chain_encoding'][truncate_index:truncate_index+self.max_length]
        item['X'] = torch.from_numpy(np.stack([item['N'], item['CA'], item['C'], item['O']], axis=1)).float()
        item['seq'] = torch.tensor(self.tokenizer.encode(item['seq']))
        return item
    

        
        
    
    def __getitem__(self, index):
        line = self.data[index]
        alphabet='ACDEFGHIKLMNPQRSTVWY'
        alphabet_set = set([a for a in alphabet])
        item = self.read_line(line, alphabet_set)

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

        data =  self._get_features(item)

        # V, E, T, T_ts, batch_id, edge_idx, chain_encoding = GeoFeaturizer.from_X_to_features(data['X'], data['edge_idx'], data['batch_id'], data['chain_encoding'], virtual_frame_num=3)
        # data['title'] = item['title']
        # data['V'] = V
        # data['E'] = E
        # data['T_ts_rots'] = T_ts._rots._rot_mats
        # data['T_ts_trans'] = T_ts._trans
        # data['batch_id_extend'] = batch_id
        # data['edge_idx_extend'] = edge_idx
        # data['chain_encoding'] = chain_encoding
        return data