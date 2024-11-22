import torch
from torch_scatter import scatter_sum, scatter_min, scatter_softmax
import torch.nn.functional as F


class GraphTransform:
    def __init__(self) -> None:
        pass

    @classmethod
    def dense2sparse_node(self, h_V, mask):
        '''
        h_V, mask --> h_V, batch_id
        '''
        B, L = h_V.shape[:2]
        shape = h_V.shape
        h_V = h_V.reshape(B,L,-1)
        node_mask_select = lambda x: torch.masked_select(x, mask.unsqueeze(-1)).reshape(-1, x.shape[-1])
        sparse_idx = mask.nonzero()
        batch_id = sparse_idx[:,0]
        h_V = node_mask_select(h_V)
        h_V = h_V.reshape(-1, *shape[2:])
        return h_V, batch_id
    
    @classmethod
    def dense2sparse_edge(self, h_E, E_idx, E_mask, num_nodes):
        '''
        h_E, E_idx, E_mask --> h_E, edge_idx
        '''
        B, L, K = h_E.shape[:3]
        shape = h_E.shape
        h_E = h_E.reshape(B,L,K,-1)
        edge_mask_select = lambda x: torch.masked_select(x, E_mask.unsqueeze(-1)).reshape(-1, x.shape[-1])
        h_E = edge_mask_select(h_E)

        # mask = E_mask.sum(dim=-1)>0

        # edge index
        # num_nodes = mask.sum(dim=1)
        N = num_nodes.max()
        shift = num_nodes.cumsum(dim=0)
        shift = torch.cat([torch.tensor([0], device=shift.device), shift[:-1]], dim=0)
        B = shift.shape[0]
        
        src = shift.view(B,1,1) + E_idx
        dst = shift.view(B,1,1) + torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(E_mask)
        src = torch.masked_select(src, E_mask).view(1,-1)
        dst = torch.masked_select(dst, E_mask).view(1,-1)
        edge_idx = torch.cat((dst, src), dim=0).long()
        h_E = h_E.reshape(-1, *shape[3:])
        return h_E, edge_idx
    
    @classmethod
    def sparse2dense_node(self, h_V, batch_id):
        '''
        h_V, batch_id --> h_V, mask
        '''
        L = h_V.shape[0]
        shape = h_V.shape
        h_V = h_V.reshape(L,-1)

        device = h_V.device
        num_nodes = scatter_sum(torch.ones_like(batch_id), batch_id)
        batch = num_nodes.shape[0]
        N = num_nodes.max()
        
        # node feature
        dim_V = torch.prod(torch.tensor(shape[1:])).item()
        h_V_dense = torch.zeros([batch, N, dim_V], device=device)
        row = batch_id
        col = torch.cat([torch.arange(0,n) for n in num_nodes]).to(device)
        h_V_dense[row, col] = h_V
        
        mask = torch.zeros([batch, N], device=device)
        mask[row, col] = 1

        h_V_dense = h_V_dense.reshape(*mask.shape[:2], *shape[1:])
        return h_V_dense, mask
    
    @classmethod
    def sparse2dense_edge(self, h_E, edge_idx, batch_id):
        '''
        h_E, edge_idx, batch_id --> h_E, E_idx, E_mask
        '''
        L = h_E.shape[0]
        shape = h_E.shape
        h_E = h_E.reshape(L,-1)

        device = h_E.device
        num_nodes = scatter_sum(torch.ones_like(batch_id), batch_id)
        batch = num_nodes.shape[0]
        N = num_nodes.max()
        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        # edge feature
        K = scatter_sum(torch.ones_like(src_idx), src_idx).max()
        dim_E = torch.prod(torch.tensor(shape[1:])).item()
        h_E_ = torch.zeros([batch, N, K, dim_E], device=device)
        row = batch_id[src_idx]
        batch_shift, _ = scatter_min(src_idx, batch_id[src_idx])
        local_dst_idx = src_idx - batch_shift[batch_id[src_idx]]
        local_src_idx = dst_idx - batch_shift[batch_id[dst_idx]]
        
        nn_num = scatter_sum(torch.ones_like(src_idx), src_idx)
        nn_idx = torch.cat([torch.arange(0,n) for n in nn_num]).to(device)
        h_E_[row, local_dst_idx, nn_idx] = h_E
        h_E = h_E_
        
        nn_num = scatter_sum(torch.ones_like(src_idx), src_idx)
        nn_idx = torch.cat([torch.arange(0,n) for n in nn_num]).to(device)
        
        E_idx = torch.arange(0, K, device=device).reshape(1,1,K).repeat(batch, N, 1)
        E_idx[row, local_dst_idx, nn_idx] = local_src_idx
        E_mask = torch.zeros([batch, N, K], device=device)
        E_mask[row, local_dst_idx, nn_idx] = 1

        h_E = h_E.reshape(*E_mask.shape[:3], *shape[1:])
        return h_E, E_idx, E_mask

    @classmethod
    def dense_cat_edge(self, h_E, h_B, num_node):
        '''
        h_E, h_, num_node --> h_E_new
        h_E: B, L, K1, ...
        h_B: B, L, K2, ...
        h_E_new:  B, L, L1+K2, ...
        '''
        B, L = h_E.shape[:2]
        dL = h_B.shape[2]
        results = []
        for b in range(B):
            temp = torch.cat([h_E[b, :num_node[b]], h_B[b,:num_node[b]]], dim=1) # 扩充K维度
            pad = torch.zeros(L+dL-num_node[b],*temp.shape[1:], device=temp.device) # padding L维度, 如果考虑old node-->new node的边,在此之前需要扩充L维度, 这里不更新global frame，因此直接padding了。
            results.append(torch.cat([temp, pad], dim=0))
        h_E_new = torch.stack(results, dim=0)
        return h_E_new

