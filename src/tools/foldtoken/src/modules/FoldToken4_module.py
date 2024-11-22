import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
import numpy as np
from src.tools.affine_utils import Rotation, Rigid, quat_to_rot, rot_to_quat, invert_rot_mat
from torch_geometric.nn.pool import knn_graph
import time
import random
from typing import Optional, Tuple

from torch_scatter import scatter_sum, scatter_max, scatter_softmax
from torch_scatter.utils import broadcast
from einops import rearrange
import pdb

from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)



def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device, dtype=values.dtype)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


def build_MLP(n_layers,dim_in, dim_hid, dim_out, dropout = 0.0, activation=nn.ReLU, normalize=True):
    if normalize:
        layers = [nn.Linear(dim_in, dim_hid), 
                nn.BatchNorm1d(dim_hid), 
                nn.Dropout(dropout), 
                activation()]
    else:
        layers = [nn.Linear(dim_in, dim_hid), 
                nn.Dropout(dropout), 
                activation()]
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(dim_hid, dim_hid))
        if normalize:
            layers.append(nn.BatchNorm1d(dim_hid))
        layers.append(nn.Dropout(dropout))
        layers.append(activation())
    layers.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*layers)

class GeoFeat(nn.Module):
    def __init__(self, geo_layer, num_hidden, virtual_atom_num, dropout=0.0):
        super(GeoFeat, self).__init__()
        self.__dict__.update(locals())
        self.virtual_atom = nn.Linear(num_hidden, virtual_atom_num*3)
        self.virtual_direct = nn.Linear(num_hidden, virtual_atom_num*3)
        # self.we_condition = build_MLP(geo_layer, 4*virtual_atom_num*3+9+16+272, num_hidden, num_hidden, dropout)
        self.we_condition = build_MLP(geo_layer, 4*virtual_atom_num*3+9+16+32, num_hidden, num_hidden, dropout)
        self.MergeEG = nn.Linear(num_hidden+num_hidden, num_hidden)

    def forward(self, h_V, h_E, T_ts, edge_idx):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        num_edge = src_idx.shape[0]
        num_atom = h_V.shape[0]

        # ==================== point cross attention =====================
        V_local = self.virtual_atom(h_V).view(num_atom,-1,3)
        V_edge = self.virtual_direct(h_E).view(num_edge,-1,3)
        Ks = torch.cat([V_edge,V_local[src_idx].view(num_edge,-1,3)], dim=1)
        Qt = T_ts.apply(Ks)
        Ks = Ks.view(num_edge,-1)
        Qt = Qt.reshape(num_edge,-1)
        V_edge = V_edge.reshape(num_edge,-1)
        quat_st = T_ts._rots._rot_mats[:, 0].reshape(num_edge, -1)


        RKs = torch.einsum('eij,enj->eni', T_ts._rots._rot_mats[:,0], V_local[src_idx].view(num_edge,-1,3))
        QRK = torch.einsum('enj,enj->en', V_local[dst_idx].view(num_edge,-1,3), RKs)

        D = rbf(T_ts._trans.norm(dim=-1), 0, 50, 16)[:,0].view(num_edge,-1)
        H = torch.cat([Ks, Qt, quat_st, D, QRK], dim=1)
        G_e = self.we_condition(H)
        h_E = self.MergeEG(torch.cat([h_E, G_e], dim=-1))
        return h_E

class PiFoldAttn(nn.Module):
    def __init__(self, attn_layer, num_hidden, num_V, num_E, dropout=0.0):
        super(PiFoldAttn, self).__init__()
        self.__dict__.update(locals())
        self.num_heads = 4
        self.W_V = nn.Sequential(nn.Linear(num_E, num_hidden),
                                nn.GELU())
                                
        self.Bias = nn.Sequential(
                                nn.Linear(2*num_V+num_E, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,self.num_heads))
        self.W_O = nn.Linear(num_hidden, num_V, bias=False)
        self.gate = nn.Linear(num_hidden, num_V)


    def forward(self, h_V, h_E, edge_idx):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        h_V_skip = h_V

        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)
        num_nodes = h_V.shape[0]
        
        w = self.Bias(torch.cat([h_V[src_idx], h_E, h_V[dst_idx]],dim=-1)).view(E, n_heads, 1) 
        attend_logits = w/np.sqrt(d) 

        V = self.W_V(h_E).view(-1,n_heads, d) 
        attend = scatter_softmax(attend_logits, index=src_idx, dim=0)
        h_V = scatter_sum(attend*V, src_idx, dim=0).view([num_nodes, -1])

        h_V_gate = F.sigmoid(self.gate(h_V))
        dh = self.W_O(h_V)*h_V_gate

        h_V = h_V_skip + dh
        return h_V


class UpdateNode(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.dense = nn.Sequential(
            nn.BatchNorm1d(num_hidden),
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden),
            nn.BatchNorm1d(num_hidden)
        )
        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden))
    
    def forward(self, h_V, batch_id):
        dh = self.dense(h_V)
        h_V = h_V + dh

        # # ============== global attn - virtual frame
        uni = batch_id.unique()
        mat = (uni[:,None] == batch_id[None]).to(h_V.dtype)
        mat = mat/mat.sum(dim=1, keepdim=True)
        c_V = mat@h_V
        # c_V = scatter_mean(h_V, batch_id, dim=0)

        h_V = h_V * F.sigmoid(self.V_MLP_g(c_V))[batch_id]
        return h_V

class UpdateEdge(nn.Module):
    def __init__(self, edge_layer, num_hidden, dropout=0.1):
        super(UpdateEdge, self).__init__()
        self.W = build_MLP(edge_layer, num_hidden*3, num_hidden, num_hidden, dropout, activation=nn.GELU, normalize=False)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.pred_quat = nn.Linear(num_hidden,8)

    def forward(self, h_V, h_E, T_ts, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_E = self.norm(h_E + self.W(h_EV))

        return h_E


class GeoFeaturizer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    @classmethod
    @torch.no_grad()
    def from_X_to_features(self, X, edge_idx, batch_id, chain_encoding, virtual_frame_num=3, atom_embed=None):
        T = Rigid.make_transform_from_reference(X[:,0].view(-1,3), X[:,1].view(-1,3), X[:,2].view(-1,3))
        T_ts = T[edge_idx[1],None].invert().compose(T[edge_idx[0],None])
        if virtual_frame_num>0:
            V, E = self.get_interact_feats(T, T_ts, X, edge_idx, batch_id, atom_embed=atom_embed)
            V_g, T_g, T_gs, edge_idx_g, batch_id_g = self.construct_virtual_frame(T, batch_id, virtual_frame_num)
            V_g = self.int_embedding(V_g, V.shape[-1])
            E_g = torch.zeros((edge_idx_g.shape[1], E.shape[1]), device=V.device, dtype=V.dtype)
            V, E, T, T_ts, batch_id, edge_idx = self.merge_local_global(V, V_g, E, E_g, T, T_g, T_ts, T_gs, batch_id, edge_idx, batch_id_g, edge_idx_g)
            chain_encoding = torch.cat([chain_encoding, torch.ones_like(batch_id_g)+1000], dim=0)
        else:
            V, E = self.get_interact_feats(T, T_ts, X, edge_idx, batch_id, atom_embed=atom_embed)
        
        
        
        return V, E, T, T_ts, batch_id, edge_idx, chain_encoding

    @classmethod
    @torch.no_grad()
    def construct_virtual_frame(self, T, batch_id, num_global=3):
        num_nodes = scatter_sum(torch.ones_like(batch_id), batch_id, dim_size=batch_id.max()+1)
            
        global_src = torch.cat([batch_id + batch_id.shape[0] +k*num_nodes.shape[0] for k in range(num_global)])
        global_dst = torch.arange(batch_id.shape[0], device=batch_id.device).repeat(num_global)
        edge_idx_g = torch.cat(
            [torch.stack([global_src, global_dst]),
            torch.stack([global_dst, global_src])],
            dim=1)
        batch_id_g = torch.arange(num_nodes.shape[0], device=batch_id.device).repeat(num_global)

        '''
        global_src: N+1,N+1,N+2,N+2,..N+B, N+B+1,N+B+1,N+B+2,N+B+2,..N+B+B
        global_dst: 0,  1,  2,  3,  ..N,   0,    1,    2,    3,    ..N
        batch_id_g: 1,  1,  2,  2,  ..B,   1,    1,    2,    2,    ..B
        '''

        X_c = T._trans

        R = torch.eye(3, device=X_c.device)[None].repeat(len(num_nodes),1,1)
        X_m = torch.zeros(len(num_nodes),3, device=X_c.device)
        rot_g = R.repeat_interleave(num_global,dim=0)
        trans_g = X_m.repeat_interleave(num_global,dim=0)
        T_g = Rigid(Rotation(rot_g), trans_g)

        rot_gs = torch.eye(3, device=X_c.device)[None,None].repeat(edge_idx_g.shape[1],1,1,1)
        trans_gs = torch.zeros(3, device=X_c.device)[None,None].repeat(edge_idx_g.shape[1],1,1)

        T_gs = Rigid(Rotation(rot_gs), trans_gs)

        h_V_g = torch.arange(num_global, device=batch_id.device).repeat(num_nodes.shape[0])
        return h_V_g, T_g, T_gs, edge_idx_g, batch_id_g

    @classmethod
    @torch.no_grad()
    def int_embedding(self, d, num_embeddings=16):
        frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=d.device)
        * -(np.log(10000.0) / num_embeddings)
        )
        angles = d[:,None] * frequency[None,:]
        angles = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return angles

    @classmethod
    @torch.no_grad()
    def positional_embeddings(self, E_idx, dtype, num_embeddings=None):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        d = E_idx[0]-E_idx[1]
        E = self.int_embedding(d, num_embeddings)
        return E


    @classmethod
    @torch.no_grad()
    def get_interact_feats(self, T, T_ts, X, edge_idx, batch_id, num_rbf=16, atom_embed=None):
        dtype = X.dtype
        device = X.device
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        num_N, num_E = X.shape[0], edge_idx.shape[1]

        def rbf_func(D, num_rbf):
            shape = D.shape
            D_min, D_max, D_count = 0., 20., num_rbf
            D_mu = torch.linspace(D_min, D_max, D_count, dtype=dtype, device=device)
            D_mu = D_mu.view([1]*(len(shape))+[-1])
            D_sigma = (D_max - D_min) / D_count
            D_expand = torch.unsqueeze(D, -1)
            RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
            return RBF

        def decouple(U):
            norm = U.norm(dim=-1, keepdim=True)
            direct = U/(norm+1e-6)
            rbf = rbf_func(norm[...,0], num_rbf)
            return torch.cat([direct, rbf], dim=-1)
        
        ## ========== new_simplified_feat_diff_vec
        diffX = F.pad(X.reshape(-1,3).diff(dim=0), (0,0,1,0)).reshape(num_N, -1, 3)
        diffX_proj = T[:,None].invert()._rots.apply(diffX)
        V = decouple(diffX_proj).reshape(num_N, -1)
        if atom_embed is not None:
            X_atom = atom_embed(torch.arange(4, device=device).repeat(num_N,1))
            dV = decouple(X_atom).reshape(num_N, -1)
            V += dV

        # V_prev = T[:,None].invert()._rots.apply(F.pad(diffX[:-1],(0,0,0,0,1,0)))
        # V_current = T[:,None].invert()._rots.apply(diffX)
        # V_next = T[:,None].invert()._rots.apply(F.pad(diffX[1:],(0,0,0,0,0,1)))
        # V_next2 = T[:,None].invert()._rots.apply(F.pad(diffX[2:],(0,0,0,0,0,2)))
        # V = torch.cat([ decouple(V_prev).reshape(num_N, -1),
        #             decouple(V_current).reshape(num_N, -1),
        #             decouple(V_next).reshape(num_N, -1),
        #             decouple(V_next2).reshape(num_N, -1)
        #             ], dim=1)
        V[torch.isnan(V)] = 0



        '''X [N,4,3]: N个氨基酸, 每个氨基酸4个原子(N,CA,C,O), 3是原子的xyz坐标
            T [N]: N个局部坐标系
        '''
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        
        diffE = T[src_idx,None].invert().apply(torch.cat([X[src_idx],X[dst_idx]], dim=1))
        diffE = decouple(diffE).reshape(num_E, -1)
        if atom_embed is not None:
            X_atom = atom_embed(torch.arange(4, device=device).repeat(num_N,1))
            E_atom = torch.cat([X_atom[src_idx],X_atom[dst_idx]], dim=1)
            dE = decouple(E_atom).reshape(num_E, -1)
            diffE += dE

        pos_embed = self.positional_embeddings(edge_idx, dtype, 16)
        # E = decouple(E).reshape(num_E,-1)
        E_quant = T_ts.invert()._rots._rot_mats.reshape(num_E,9)
        E_trans = T_ts._trans
        E_trans = decouple(E_trans).reshape(num_E,-1)
        E = torch.cat([diffE, E_quant, E_trans, pos_embed], dim=-1)
        return V, E

    @classmethod
    @torch.no_grad()
    def merge_local_global(self, h_V, h_V_g, h_E, h_E_g, T, T_g, T_ts, T_gs, batch_id, edge_idx, batch_id_g, edge_idx_g):
        h_V = torch.cat([h_V, h_V_g], dim=0)
        h_E = torch.cat([h_E, h_E_g], dim=0)
        T = T.cat([T, T_g], dim=0)

        batch_id = torch.cat([batch_id, batch_id_g])
        edge_idx = torch.cat([edge_idx, edge_idx_g], dim = -1)
        T_ts = T_ts.cat([T_ts, T_gs], dim = 0)
        return h_V, h_E, T, T_ts, batch_id, edge_idx



class GeneralGNN(nn.Module):
    def __init__(self, 
                 geo_layer, 
                 attn_layer,
                 ffn_layer,
                 edge_layer,
                 num_hidden, 
                 virtual_atom_num=32, 
                 dropout=0.1,
                 mask_rate=0.15):
        super(GeneralGNN, self).__init__()
        self.__dict__.update(locals())
        self.geofeat = GeoFeat(geo_layer, num_hidden, virtual_atom_num, dropout)
        self.attention = PiFoldAttn(attn_layer, num_hidden, num_hidden, num_hidden, dropout) 
        self.update_node = UpdateNode(num_hidden)
        self.update_edge = UpdateEdge(edge_layer, num_hidden, dropout)
        self.mask_token = nn.Embedding(2, num_hidden)
    
        
    def forward(self, h_V, h_E, T_ts, edge_idx, batch_id):

        h_E = self.geofeat(h_V, h_E, T_ts, edge_idx)
        h_V = self.attention(h_V, h_E, edge_idx)
        h_V = self.update_node(h_V, batch_id)
        h_E = self.update_edge( h_V, h_E, T_ts, edge_idx, batch_id )
        return h_V, h_E


class StructureEncoder(nn.Module):
    def __init__(self, 
                 geo_layer, 
                 attn_layer,
                 ffn_layer,
                 edge_layer, 
                 encoder_layer,
                 hidden_dim, 
                 dropout=0,
                 version='CATH-AE'):
        """ Graph labeling network """
        super(StructureEncoder, self).__init__()
        self.__dict__.update(locals())


        self.node_embedding = build_MLP(2, 76, hidden_dim, hidden_dim)
        self.edge_embedding = build_MLP(2, 196, hidden_dim, hidden_dim)
        self.virtual_embedding = nn.Embedding(30, hidden_dim)
        self.chain_embedding = nn.Embedding(2, hidden_dim)
        # self.temp_embedding = nn.Linear(1, hidden_dim)

        self.encoder_layers = nn.ModuleList([GeneralGNN(geo_layer, 
                 attn_layer,
                 ffn_layer,
                 edge_layer, 
                 hidden_dim, 
                 dropout=dropout) for i in range(encoder_layer)])
        self.s = nn.Linear(hidden_dim, 1)
    
    
    
    def decouple_local_global(self, h_V, h_E, batch_id, edge_idx, h_V_g, h_E_g, batch_id_g):
        # ============== 解耦合local&global edges
        num_node_g = batch_id_g.shape[0]
        num_edge_g = h_E_g.shape[0]
        batch_id, batch_id_g = batch_id[:-num_node_g], batch_id[-num_node_g]
        h_V, h_V_g = h_V[:-num_node_g], h_V[-num_node_g:]
        h_E, h_E_g = h_E[:-num_edge_g], h_E[-num_edge_g:]
        edge_idx, global_edge = edge_idx[:,:-num_edge_g], edge_idx[:,-num_edge_g:]
        return h_V, h_E, h_V_g, h_E_g

    

    def forward(self, X, edge_idx, batch_id, chain_encoding, V=None, E=None, T_ts=None, batch_id_extend=None, edge_idx_extend=None, virtual_frame_num=3, temperature=1.0):
        N = batch_id.shape[0]
        if V is None: # for sampling, initial V, E, T_ts
            V, E, T, T_ts, batch_id, edge_idx, chain_encoding = GeoFeaturizer.from_X_to_features(X, edge_idx, batch_id, chain_encoding, virtual_frame_num)
        else:
            batch_id = batch_id_extend
            edge_idx = edge_idx_extend
        
        h_V, h_E = self.node_embedding(V), self.edge_embedding(E) + self.chain_embedding((chain_encoding[edge_idx[0]]==chain_encoding[edge_idx[1]]).long())
        
        # h_V = h_V+self.temp_embedding(torch.tensor(temperature, device=h_V.device)[None,None])
        # if self.version == 'PDB-VQ':
        #     h_V, h_E = self.node_embedding(V), self.edge_embedding(E) + self.chain_embedding((edge_idx[0]==edge_idx[1]).long()) # there is a bug in chain encoding
        
        # if self.version == 'CATH-AE':
        #     h_V, h_E = self.node_embedding(V), self.edge_embedding(E) + self.chain_embedding((chain_encoding[edge_idx[0]]==chain_encoding[edge_idx[1]]).long())
        
        # if self.version == 'CATH-VQ':
        #     h_V, h_E = self.node_embedding(V), self.edge_embedding(E)

        outputs = []
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, T_ts, edge_idx, batch_id)
            outputs.append(h_V.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        S = F.sigmoid(self.s(outputs))
        output = torch.einsum('nkc, nkb -> nbc', outputs, S).squeeze(1)
        return output
    
class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab=33):
        super().__init__()
        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V):
        logits = self.readout(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits


class GeneralE3GNN(nn.Module):
    def __init__(self, gnn_layers, num_hidden, version='CATH-AE'):
        super(GeneralE3GNN, self).__init__()
        self.__dict__.update(locals())
        self.node_embedding = build_MLP(2, 76, num_hidden, num_hidden)
        self.edge_embedding = build_MLP(2, 196, num_hidden, num_hidden)
        self.virtual_embedding = nn.Embedding(30, num_hidden) 
        self.chain_embedding = nn.Embedding(2, num_hidden)
        self.gnns = nn.ModuleList([GeneralGNN(3,3,3,3,num_hidden,32,0.0) for _ in range(gnn_layers)])

        self.W_logit = nn.Linear(num_hidden, 2)
        self.W_q = nn.Linear(num_hidden, 9)
        self.W_t = nn.Linear(num_hidden, 3)
        self.W_s_edge = nn.Linear(num_hidden, 2)
        self.W_atom = nn.Linear(num_hidden, 12)
        self.V_feat = nn.Linear(30, num_hidden)
        self.E_feat = nn.Linear(25, num_hidden)
        self.V_norm = nn.BatchNorm1d(num_hidden)
        self.rotmat2quat = nn.Linear(9,4)
        # self.atom_embed = nn.Embedding(4,3)
        self.atom_embed = None
    
    def forward(self, X, h_V, batch_id, chain_encoding, topk=30, virtual_frame_num=3):
        N = X.shape[0]
        edge_idx = self.build_graph(X, batch_id, topk)
        dV, dE, T, T_ts, batch_id, edge_idx, chain_encoding = GeoFeaturizer.from_X_to_features(X, edge_idx, batch_id, chain_encoding, virtual_frame_num, self.atom_embed)
        dV = self.node_embedding(dV)

        dE = self.edge_embedding(dE) + self.chain_embedding((chain_encoding[edge_idx[0]]==chain_encoding[edge_idx[1]]).long())

        # if self.version == 'CATH-VQ':
        #     dE = self.edge_embedding(dE)
        # if self.version == 'CATH-AE' or self.version == 'PDB-VQ':
        #     dE = self.edge_embedding(dE) + self.chain_embedding((edge_idx[0]==edge_idx[1]).long()) 

        if h_V.shape[0]!=dV.shape[0]:
            h_V = F.pad(h_V, (0,0, 0, dV.shape[0]-h_V.shape[0]))
        h_V = self.V_norm(h_V+dV)
        h_E = dE
        # hE变化了
        for i, gnn in enumerate(self.gnns):
            h_V, h_E = gnn(h_V, h_E, T_ts, edge_idx, batch_id)

        R_s, t_s = self.message_passing_R_mat(T, T_ts, h_E, edge_idx, batch_id)
        
        T = Rigid(Rotation(R_s), t_s)
        self.local_atoms = self.W_atom(h_V).view(-1, 4, 3)
        dX = T[:,None].apply(self.local_atoms)
        return dX[:N], h_V
        # return (X + dX[:N])/2, h_V # add x shortcut
    
    
    
    

    def message_passing_R_mat(self, T, T_ts, h_E, edge_idx, batch_id):
        # quat_ts_init = rot_to_quat(T_ts._rots._rot_mats)[:, 0]
        quat_ts_init = T_ts._rots._rot_mats[:, 0].reshape(-1,9)
        t_ts_init = T_ts._trans[:, 0]
        all_num_node = batch_id.shape[0]
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        s_edge = torch.sigmoid(self.W_s_edge(h_E)[..., None]).unbind(-2)
        d_scale = 10
        q_ts = s_edge[0] * quat_ts_init + (1.0 - s_edge[0]) * self.W_q(h_E)
        # q_ts = q_ts/(q_ts.norm(dim=-1, keepdim=True)+1e-6)
        t_ts = s_edge[1] * t_ts_init + (1.0 - s_edge[1]) * d_scale * self.W_t(h_E)
        # R_ts = quat_to_rot(q_ts)
        R_ts = self.avg_rotation(q_ts.reshape(-1,3,3))
        # R_ts = quat_to_rot(q_ts)

        logit_ts = self.W_logit(h_E)
        R_s = T._rots._rot_mats
        t_s = T._trans
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        R_s, t_s = self.equilibrate_transforms(R_s, t_s,
                                        R_ts, t_ts,
                                        logit_ts,
                                        src_idx, dst_idx,
                                        all_num_node)
        return R_s, t_s

    def equilibrate_transforms(
        self,
        R_i_init, t_i_init,
        R_ji, t_ji,
        logit_ji,
        src_idx, dst_idx,
        all_num_node):
        R_i_pred, t_i_pred = self.compose_transforms(R_i_init[dst_idx], t_i_init[dst_idx], R_ji, t_ji)

        probs = softmax(logit_ji, src_idx, dim=0, num_nodes=src_idx.max()+1)
        t_probs, R_probs = probs.unbind(-1)
        L = all_num_node
        t_avg = scatter_sum(t_i_pred*t_probs[:,None], src_idx, dim=0, dim_size=L)
        R_avg_unc = scatter_sum(R_i_pred*R_probs[:,None,None], src_idx, dim=0, dim_size=L)
        R_avg = self.avg_rotation(R_avg_unc)
        

        return R_avg, t_avg

    def avg_rotation(self, R_avg_unc, dither_eps = 1e-4):
        # Q, R = torch.qr(R_avg_unc)
        # return Q

        # # ==================== SVD projection
        # R_avg_unc2 = R_avg_unc + dither_eps * torch.randn_like(R_avg_unc)
        # U, S, Vh = torch.linalg.svd(R_avg_unc2.float(), full_matrices=True)
        # idx = 0
        # while not self.svd_is_stable(S):
        #     R_avg_unc2 = R_avg_unc + dither_eps * torch.randn_like(R_avg_unc)
        #     U, S, Vh = torch.linalg.svd(R_avg_unc2.float(), full_matrices=True)
        #     idx += 1
        #     if idx>10:
        #         print('SVD is consistently unstable')
        #         break
        # # U, S, Vh = svd(R_avg_unc.float())
        # R_avg = U @ Vh

        # # Enforce that matrix is rotation matrix
        # d = torch.linalg.det(R_avg)
        # d_expand = F.pad(d[..., None, None], (2, 0), value=1.0)
        # Vh = Vh * d_expand
        # R_avg = U @ Vh
        
        # ===================== quat projection
        quat_avg = self.rotmat2quat(R_avg_unc.reshape(-1,9))
        quat_avg = quat_avg/(quat_avg.norm(dim=-1, keepdim=True)+1e-6)
        R_avg = quat_to_rot(quat_avg)
        return R_avg



    def compose_transforms(self,
        R_a: torch.Tensor, t_a: torch.Tensor, R_b: torch.Tensor, t_b: torch.Tensor
    ):
        """Compose transforms `T_compose = T_a * T_b` (broadcastable).

        Args:
            R_a (torch.Tensor): Transform `T_a` rotation matrix with shape `(...,3,3)`.
            t_a (torch.Tensor): Transform `T_a` translation with shape `(...,3)`.
            R_b (torch.Tensor): Transform `T_b` rotation matrix with shape `(...,3,3)`.
            t_b (torch.Tensor): Transform `T_b` translation with shape `(...,3)`.

        Returns:
            R_composed (torch.Tensor): Composed transform `a * b` rotation matrix with
                shape `(...,3,3)`.
            t_composed (torch.Tensor): Composed transform `a * b` translation vector with
                shape `(...,3)`.
        """
        R_composed = R_a @ R_b
        t_composed = t_a + (R_a @ t_b.unsqueeze(-1)).squeeze(-1)
        return R_composed, t_composed
    
    @classmethod
    def build_graph(self, X, batch_id, topk=30):
        X = X[:,1,:]
        edge_idx = knn_graph(X, k=topk, batch=batch_id, loop=False, flow='target_to_source')     

        ### batch化计算distance的时候，我每个batch做了一下shift偏移，当偏移量大的时候，原始数据的值相对太小，被忽略了。在batch太大的时候，这个问题会更加明显
        # shift = batch_id[:,None]*X.abs().max()*2
        # X = X+shift

        # distances = torch.sum(X ** 2, dim=1, keepdim=True) \
        #           + torch.sum(X ** 2, dim=1) \
        #           - 2 * torch.matmul(X, X.t())
        # _, dst_idx = torch.topk(-distances, min(distances.shape[1],topk))
        # src_idx = dst_idx[:,0:1].expand_as(dst_idx)
        # edge_idx = torch.stack([src_idx[:,1:].reshape(-1), dst_idx[:,1:].reshape(-1)])
        return edge_idx



class StructDecoder(nn.Module):
    def __init__(self, num_decoder_layers, hidden_dim, steps=5, version='CATH-AE'):
        super().__init__()
        self.steps = steps
        self.embed_condition = nn.Linear(hidden_dim, hidden_dim)
        self.embed_score = nn.Linear(1, hidden_dim)
        self.decoders = nn.ModuleList(
            nn.ModuleList([GeneralE3GNN(1, hidden_dim, version=version) for i in range(num_decoder_layers)])
            for i in range(steps)
        )
        


    def infer_X(self, X, h_V_enc, batch_id, chain_encoding, dec_topk=30, virtual_frame_num=3):
        # torch.save(X, '/huyuqi/xmyu/FoldToken2/X_32.pt')
        # X[-4012:] = torch.load('/huyuqi/xmyu/FoldToken2/X_8.pt')
        step = 0
        h_V = self.embed_condition(h_V_enc) #+ self.embed_score(score)[batch_id]
        all_preds = []
        for layer in self.decoders[step]:
            X, _ = layer(X.detach(), h_V, batch_id, chain_encoding, dec_topk, virtual_frame_num=virtual_frame_num)
            all_preds.append(X)
        return X, all_preds
    
    # def train_multi_steps(self, X, h_V_enc, batch_id, chain_encoding, dec_topk=30, virtual_frame_num=3):
    #     step = 0
    #     X, all_preds = self.infer_one_step(X, h_V_enc, batch_id, chain_encoding, step, dec_topk, virtual_frame_num=virtual_frame_num)
    #     return X, all_preds

    # def infer_multi_steps(self, X, h_V_enc, batch_id, chain_encoding, dec_topk=30, virtual_frame_num=3):
    #     step = 0
    #     with torch.no_grad():
    #         X, _ = self.infer_one_step(X, h_V_enc, batch_id, chain_encoding, step, dec_topk, virtual_frame_num=virtual_frame_num)
    #     return X




