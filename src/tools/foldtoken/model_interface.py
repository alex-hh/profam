import inspect
import torch
import torch.nn as nn
import os
from src.interface.model_interface import MInterface_base
from omegaconf import OmegaConf
from src.modules.pifold_module_sae import GeoFeaturizer
from torch_geometric.nn import knn_graph
from src.chroma.data import Protein
from src.tools.geo_utils import batch_rmsd

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.cross_entropy = nn.NLLLoss(reduction='none')

        self.train_steps_per_epoch = self.hparams.steps_per_epoch
        self.temp_scale = 0.001
        self.vqshortcut = True
    

    def forward(self, batch, batch_idx, mode='train', level=8):
        V, E, T, T_ts, batch_id_extend, edge_idx_extend, chain_encoding = GeoFeaturizer.from_X_to_features(batch['X'], batch['edge_idx'], batch['batch_id'], batch['chain_encoding'], virtual_frame_num=3)

        X, edge_idx, batch_id = batch['X'], batch['edge_idx'], batch['batch_id']
        X_t = torch.rand_like(X)
        temp = batch['temp']

        X_pred, all_preds, vq_loss, vq_count, vq_code = self.model(X, chain_encoding,
                                    edge_idx, batch_id, 
                                    X_t, V, E, T_ts, 
                                    temp, mode, 
                                    batch_id_extend = batch_id_extend,
                                    edge_idx_extend = edge_idx_extend,
                                    virtual_frame_num=self.hparams.virtual_frame_num,
                                    vqshortcut = self.vqshortcut,
                                    level = level)
        
        return {'all_preds': all_preds, 'vq_loss':vq_loss, 'vq_count':vq_count, 'vq_code': vq_code} 

    
    def load_model(self):
        from src.models.FoldToken4 import PiFold_Model
        self.model = PiFold_Model(self.hparams)


    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
    
    def batch_data_single_chain(self, list_of_data):
        name_all = []
        batch_id_all = []
        chain_encoding_all = []
        X_all = []
        S_all = []
        idx = 0
        for (name, X, C, S) in list_of_data:
            batch_id = torch.zeros_like(S)+idx
            chain_encoding = C
            name_all.append(name)
            batch_id_all.append(batch_id)
            chain_encoding_all.append(chain_encoding)
            X_all.append(X)
            S_all.append(S)
            idx += 1
        
        S_all = torch.cat(S_all, dim=0)
        X_all = torch.cat(X_all, dim=0)
        batch_id_all = torch.cat(batch_id_all, dim=0)
        chain_encoding_all = torch.cat(chain_encoding_all, dim=0)
        edge_idx = knn_graph(X_all[:,1], k=30, batch=batch_id_all, loop=True, flow='target_to_source')
        return {'title':name_all ,'S':S_all.cuda(), 'X': X_all.cuda(), 'edge_idx': edge_idx.cuda(), 'batch_id': batch_id_all.cuda(), 'chain_encoding': chain_encoding_all.cuda(), 'temp': 1e-8, 'mode': 'eval'}
    
    
    def batch_proteins(self, list_of_proteins):
        name_all = []
        batch_id_all = []
        chain_encoding_all = []
        X_all = []
        S_all = []
        for idx, protein in enumerate(list_of_proteins):
            # name = protein.sys.name.split('/')[-1]
            X, C, S = protein.to_XCS()
            X = X[C>0][None,]
            S = S[C>0][None,]
            C = C[C>0][None,]
            L = X.shape[1]
            batch_id = torch.zeros_like(S)[0]+idx
            chain_encoding = C[0]
            # name_all.append(name)
            batch_id_all.append(batch_id)
            chain_encoding_all.append(chain_encoding)
            X_all.append(X[0])
            S_all.append(S[0])
        
        S_all = torch.cat(S_all, dim=0)
        X_all = torch.cat(X_all, dim=0)
        batch_id_all = torch.cat(batch_id_all, dim=0)
        chain_encoding_all = torch.cat(chain_encoding_all, dim=0)
        edge_idx = knn_graph(X_all[:,1], k=30, batch=batch_id_all, loop=True, flow='target_to_source')
        return {'S':S_all, 'X': X_all, 'edge_idx': edge_idx, 'batch_id': batch_id_all, 'chain_encoding': chain_encoding_all, 'temp': 1e-8, 'mode': 'eval'}
    
    def encode_only(self, batch, level=8):
        self.eval()
        V, E, T, T_ts, batch_id_extend, edge_idx_extend, chain_encoding = GeoFeaturizer.from_X_to_features(batch['X'], batch['edge_idx'], batch['batch_id'], batch['chain_encoding'], virtual_frame_num=3)

        X, edge_idx, batch_id = batch['X'], batch['edge_idx'], batch['batch_id']
        X_t = torch.rand_like(X)
        temp = batch['temp']

        h_V = self.model.encoder(X, edge_idx, batch_id, chain_encoding, V, E, T_ts, batch_id_extend, edge_idx_extend, virtual_frame_num=3, temperature=temp)
                
        h_V = h_V[chain_encoding<1000]
        h_V, vq_code, vq_loss = self.model.vq(h_V, temperature=temp, mode='eval', vqshortcut=False, level=level)
        vq_code_all = []
        
        for i in range(batch_id.max()+1):
            vq_code_all.append(vq_code[batch_id==i])
            
        return vq_code_all

    def sample(self,  batch, level=8, cal_rmsd=False):
        self.eval()
        ret = self(batch, 0, mode='eval', level=level)
        X_pred = ret['all_preds'][-1]
        batch_id = batch['batch_id']
        
        vq_code_all = []
        protein_all = []
        
        for i in range(batch_id.max()+1):
            X = X_pred[batch_id==i]
            C = batch['chain_encoding'][batch_id==i]
            S = torch.ones_like(C)
            protein = Protein.from_XCS(X[None], C[None], S[None])
            protein_all.append(protein)
            vq_code_all.append(ret['vq_code'][batch_id==i])
        
        if cal_rmsd:
            rmsd = batch_rmsd(X_pred[:,1], batch['X'][:,1], batch_id)
            return protein_all, vq_code_all, rmsd
        return protein_all, vq_code_all
    
    def encode_protein(self, protein_init, level=8, Ablation=False):
        self.eval()
        X, C, S = protein_init.to_XCS()
        X = X[C>0][None,]
        S = S[C>0][None,]
        C = C[C>0][None,]
        
        batch_id = torch.zeros_like(S)[0]
        X = X[0]
        edge_idx = knn_graph(X[:,1], k=30, batch=batch_id, loop=True, flow='target_to_source')
        chain_encoding = torch.zeros_like(batch_id)+1

        batch = {'X': X, 'edge_idx': edge_idx, 'batch_id': batch_id, 'chain_encoding': chain_encoding, 'temp': 1e-8, 'mode': 'eval'}
        
        V, E, T, T_ts, batch_id_extend, edge_idx_extend, chain_encoding = GeoFeaturizer.from_X_to_features(batch['X'], batch['edge_idx'], batch['batch_id'], batch['chain_encoding'], virtual_frame_num=3)

        X, edge_idx, batch_id = batch['X'], batch['edge_idx'], batch['batch_id']

        _, vq_code, vq_loss = self.model.encoding(X, chain_encoding, edge_idx, batch_id, V, E, T_ts, batch['temp'], batch['mode'], batch_id_extend, edge_idx_extend, virtual_frame_num=3, vqshortcut=False, level=level)
        chain_encoding = chain_encoding[chain_encoding<1000]
        
        h_V_quat = self.model.vq.embed_id(vq_code, level)
        # protein_pred = self.model.decoding(h_V_quat, chain_encoding, batch_id=None, returnX=False)
        
        return h_V_quat, vq_code,  batch_id, chain_encoding
    

