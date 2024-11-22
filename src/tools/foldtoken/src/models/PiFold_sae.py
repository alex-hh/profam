import time
import torch
import math
import torch.nn as nn
import numpy as np
from src.modules.pifold_module_sae import *
from src.chroma.layers.structure.diffusion import DiffusionChainCov, ReconstructionLosses
from src.modules.graph_transform import GraphTransform
from src.chroma.data import Protein
from src.modules.vq_modules import SoftCVQLayer
from einops import rearrange
from src.tools.geo_utils import batch_rmsd
from torch_geometric.nn import knn_graph

pair_lst = ['N-N', 'C-C', 'O-O', 'Cb-Cb', 'Ca-N', 'Ca-C', 'Ca-O', 'Ca-Cb', 'N-C', 'N-O', 'N-Cb', 'Cb-C', 'Cb-O', 'O-C', 'N-Ca', 'C-Ca', 'O-Ca', 'Cb-Ca', 'C-N', 'O-N', 'Cb-N', 'C-Cb', 'O-Cb', 'C-O']

def cross(u, v, dim=-1):
    dtype = u.dtype
    if dtype!=torch.float32:
        u = u.to(dtype=torch.float32)
        v = v.to(dtype=torch.float32)
    return torch.cross(u,v, dim).to(dtype=dtype)

class PiFold_Model(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(PiFold_Model, self).__init__()
        self.args = args
        hidden_dim = args.hidden_dim
        self.top_k = args.k_neighbors

        self.vq = SoftCVQLayer(args.vq_space, hidden_dim, 32, 6)
        self.encoder = StructureEncoder(3, 3, 3, 3, args.enc_layers, hidden_dim, dropout=0.0)
        
        self.virtual_embedding = nn.Embedding(30, hidden_dim) 
        steps = 1
        self.decoder_struct = StructDecoder(args.dec_layers, hidden_dim, steps)
        self.score_loss = nn.CrossEntropyLoss()

        self._init_params()
        self.noise_perturb = DiffusionChainCov(
                noise_schedule='log_snr',
                beta_min=0.2,
                beta_max=70,
                log_snr_range=[-7.0, 13.5],
                covariance_model='globular',
                complex_scaling=True,
            )
        

        self.loss_diffusion = ReconstructionLosses(
            diffusion=self.noise_perturb, rmsd_method='symeig', loss_scale=10.0
        )
        
    def _init_params(self):
        for name, p in self.named_parameters():
            if name == 'virtual_atoms':
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    

    def compute_loss(self, all_preds, X, batch_id):
        L = len(all_preds)
        X, mask = GraphTransform.sparse2dense_node(X, batch_id)
        all_results = {'batch_elbo':0,
                        'batch_global_mse':0,
                        'batch_fragment_mse':0,
                        'batch_pair_mse':0,
                        'batch_neighborhood_mse':0,
                        'batch_distance_mse':0}
        L = len(all_preds)
        t = torch.ones(X.shape[0], device=X.device)
        for pred_X in all_preds:
            X_pred_tmp, _ = GraphTransform.sparse2dense_node(pred_X, batch_id)
            results = self.loss_diffusion(X_pred_tmp, X, mask, t)
            
            all_results['batch_elbo'] += results['batch_elbo']/L
            all_results['batch_global_mse'] += results['batch_global_mse']/L
            all_results['batch_fragment_mse'] += results['batch_fragment_mse']/L
            all_results['batch_pair_mse'] += results['batch_pair_mse']/L
            all_results['batch_neighborhood_mse'] += results['batch_neighborhood_mse']/L
            all_results['batch_distance_mse'] += results['batch_distance_mse']/L
        pred_X, mask = GraphTransform.sparse2dense_node(pred_X, batch_id)

        loss = all_results['batch_global_mse'] + all_results['batch_fragment_mse'] + all_results['batch_pair_mse'] + all_results['batch_neighborhood_mse'] + all_results['batch_distance_mse']
        all_results['loss'] = loss


        return all_results
    

    def forward(self, X_true, chain_encoding, edge_idx, batch_id, X_t, V=None, E=None, T_ts=None, temp=1.0, mode='train', return_predX=False, batch_id_extend=None, edge_idx_extend=None, virtual_frame_num=3, vqshortcut=False, frozen_vq=0.0):
        if frozen_vq:
            with torch.no_grad():
                h_V = self.encoder(X_true, edge_idx, batch_id, chain_encoding, V, E, T_ts, batch_id_extend, edge_idx_extend, virtual_frame_num=virtual_frame_num, temperature=temp)
                    
                h_V = h_V[chain_encoding<1000]
                h_V, vq_code, vq_loss = self.vq(h_V, temperature=temp, mode=mode, vqshortcut=vqshortcut, frozen=False)
        else:
            h_V = self.encoder(X_true, edge_idx, batch_id, chain_encoding, V, E, T_ts, batch_id_extend, edge_idx_extend, virtual_frame_num=virtual_frame_num, temperature=temp)
                    
            h_V = h_V[chain_encoding<1000]
            h_V, vq_code, vq_loss = self.vq(h_V, temperature=temp, mode=mode, vqshortcut=vqshortcut, frozen=False)
        
        X_pred, all_preds= self.decoder_struct.infer_X(X_t, h_V,  batch_id, chain_encoding, self.args.dec_topk, virtual_frame_num=virtual_frame_num)
        
        vq_count = torch.bincount(vq_code, minlength=self.vq.embedding.shape[0])
        if return_predX:
            return X_pred, vq_code
        return X_pred, all_preds, vq_loss, vq_count


    def sample(self, 
               protein_init):
        self.eval()
        X, C, S = protein_init.to_XCS()
        X = X[C>0][None,]
        S = S[C>0][None,]
        C = C[C>0][None,]
        
        batch_id = torch.zeros_like(S)[0]
        X = X[0]
        edge_idx = knn_graph(X[:,1], k=30, batch=batch_id, loop=True, flow='target_to_source')
        chain_encoding = torch.cat([torch.zeros_like(batch_id), torch.ones(3, device=batch_id.device).long()*1001])
        batch = {'X_true':X, 'edge_idx':edge_idx, 'batch_id':batch_id, 'X_t': torch.rand_like(X), 'temp':1e-8, 'mode': 'eval', 'chain_encoding':chain_encoding, 'virtual_frame_num':3}

        X_pred, vq_ids = self(**batch, return_predX=True)
        X_pred = X_pred[None,...]
        protein = Protein.from_XCS(X_pred, C, S)

        return protein
    
    # def encoding(self, protein_init, multi_chain=False):
    #     self.eval()
    #     X, C, S = protein_init.to_XCS()
    #     X = X[C>0][None,]
    #     S = S[C>0][None,]
    #     C = C[C>0][None,]
        
    #     batch_id = torch.zeros_like(S)[0]
    #     X = X[0]
    #     edge_idx = knn_graph(X[:,1], k=30, batch=batch_id, loop=True, flow='target_to_source')
    #     if multi_chain:
    #         chain_encoding = torch.cat([C[0], torch.ones(3, device=batch_id.device).long()*1001])
    #     else:
    #         chain_encoding = torch.cat([torch.ones_like(batch_id), torch.ones(3, device=batch_id.device).long()*1001])


    #     h_V = self.encoder(X, edge_idx, batch_id, chain_encoding, V=None, E=None, T_ts=None, batch_id_extend=None, edge_idx_extend=None, virtual_frame_num=3)

    #     h_V = h_V[chain_encoding<1000]
    #     chain_encoding = chain_encoding[chain_encoding<1000]
    #     return h_V, chain_encoding
    
    def encoding(self, X_true, chain_encoding, edge_idx, batch_id, V=None, E=None, T_ts=None, temp=1.0, mode='train', batch_id_extend=None, edge_idx_extend=None, virtual_frame_num=3, vqshortcut=False, level=8):
        h_V = self.encoder(X_true, edge_idx, batch_id, chain_encoding, V, E, T_ts, batch_id_extend, edge_idx_extend, virtual_frame_num=virtual_frame_num, temperature=temp)
                
        h_V = h_V[chain_encoding<1000]
        h_V, vq_code, vq_loss = self.vq(h_V, temperature=temp, mode=mode, vqshortcut=vqshortcut)
        return h_V, vq_code, vq_loss

    
    def decoding(self, h_V, chain_encoding, batch_id=None, returnX=False):
        self.eval()
        L, _ = h_V.shape
        X_t = torch.rand(L,4,3, device='cuda')
        if batch_id is None:
            batch_id = torch.zeros(L, device='cuda').long()
        virtual_frame_num = 3
        
        X_pred, all_preds = self.decoder_struct.infer_X(X_t, h_V,  batch_id, chain_encoding, self.args.dec_topk, virtual_frame_num=virtual_frame_num)

        C = S = chain_encoding[None]
        X_pred = X_pred[None]
        if returnX:
            return X_pred, all_preds
        protein = Protein.from_XCS(X_pred, C, S)
        return protein


def count_parameters_in_millions(model: nn.Module) -> float:
    """
    Count the number of parameters in a PyTorch model and return the count in millions (M).
    
    Args:
    model (nn.Module): The PyTorch model for which to count parameters.
    
    Returns:
    float: The total number of parameters in millions.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6