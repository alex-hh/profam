import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from torch.autograd import Function
import numpy as np
import pdb

def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


# class GumbelSoftmax(Function):
#     @staticmethod
#     def forward(ctx, logits, tau, dim=-1):
#         y = F.softmax(logits/tau, dim)
#         ctx.save_for_backward(y)
#         ctx.tau = tau
#         return y

#     @staticmethod
#     def backward(ctx, grad_output):
#         softmax_output, = ctx.saved_tensors
#         tau = ctx.tau
#         scaling_fact = 1/max(tau, 1e-3)
#         grad_input = softmax_output * (grad_output - (grad_output * softmax_output).sum(dim=-1, keepdim=True)) * scaling_fact

#         return grad_input, None

# def gumbel_softmax(logits, tau):
#     return GumbelSoftmax.apply(logits, tau)

class StableAttn(Function):
    @staticmethod
    def forward(ctx, H, C, tau):
        distances = - 2 * torch.matmul(H, C.t())/tau
        ctx.save_for_backward(H, C)
        ctx.tau = tau
        return distances

    @staticmethod
    def backward(ctx, grad_output):
        H, C = ctx.saved_tensors
        tau = ctx.tau
        # Compute gradients
        grad_H = -2 * torch.matmul(grad_output, C)/tau
        grad_C = -2 * torch.matmul(grad_output.t(), H)
        # print(tau)

        return grad_H, grad_C, None

# def gumbel_softmax(logits, tau):
#     return StableAttn.apply(logits, tau)

class VQLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, vq_dim, ema_k, ema_lambda, ema_weight):
        super(VQLayer, self).__init__()
        self.init = True
        self.ema_k = ema_k
        self.ema_lambda = ema_lambda
        self.ema_weight = ema_weight
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, vq_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.register_buffer('count', torch.zeros(num_embeddings))
        self.MSE = nn.HuberLoss(reduction='none')
        self.proj = nn.Linear(embedding_dim, vq_dim)
        self.proj_inv = nn.Linear(vq_dim, embedding_dim)
    
    def EMAUpdate(self, h_flat, distances, k=5, gamma=0.995):
        # move toward data distribution
        dist, topk_ids = torch.topk(-distances,k=k, dim=0)
        h_feat = h_flat[topk_ids.view(-1)].reshape(k,self.num_embeddings,-1).mean(dim=0)
        
        update_mask = torch.abs(dist).min(dim=0)[0]>0.1
        updated_embedding = self.embedding.weight.data*gamma + h_feat*(1-gamma)
        self.embedding.weight.data = self.embedding.weight.data*(~update_mask[:,None]) + updated_embedding*update_mask[:,None]
        

    def embed_id(self, vq_id):
        self.embedding.weight.data = self.embedding.weight.data/self.embedding.weight.data.norm(dim=-1, keepdim=True)
        quantized = self.proj_inv(self.embedding(vq_id))
        return quantized

    def get_vq(self, h, attn_mask=None, mode='train', temperature=None):
        h = self.proj(h)
        h = h/torch.norm(h, dim=-1, keepdim=True)
        # 投影到离散空间
        h_shape = h.shape
        h_flat = h.view(-1, h_shape[-1])
            
        self.embedding.weight.data = self.embedding.weight.data/self.embedding.weight.data.norm(dim=-1, keepdim=True)
        
        distances = torch.sum(h_flat ** 2, dim=1, keepdim=True) \
                  + torch.sum(self.embedding.weight ** 2, dim=1) \
                  - 2 * torch.matmul(h_flat, self.embedding.weight.t())
        
        if attn_mask is not None:
            distances = distances*attn_mask.view(-1,1) +10000000*(1-attn_mask.view(-1,1))

        
        code = torch.argmin(distances, dim=1).view(h_shape[:-1])

        quantized = self.embed_id(code)
        return code, quantized
    
    def forward(self, h, attn_mask=None, mode='train', temperature=None):
        h = self.proj(h)
        h = h/torch.norm(h, dim=-1, keepdim=True)
        # 投影到离散空间
        h_shape = h.shape
        h_flat = h.view(-1, h_shape[-1])
            
        self.embedding.weight.data = self.embedding.weight.data/self.embedding.weight.data.norm(dim=-1, keepdim=True)
        
        distances = torch.sum(h_flat ** 2, dim=1, keepdim=True) \
                  + torch.sum(self.embedding.weight ** 2, dim=1) \
                  - 2 * torch.matmul(h_flat, self.embedding.weight.t())
        
        if attn_mask is not None:
            distances = distances*attn_mask.view(-1,1) +10000000*(1-attn_mask.view(-1,1))

        
        code = torch.argmin(distances, dim=1).view(h_shape[:-1])
        quantized = F.embedding(code, self.embedding.weight)
        quantized = h + (quantized - h).detach()
        
        
        unique_values, unique_indices = torch.unique(code.view(-1), return_inverse=True)
        counts = torch.bincount(unique_indices)
        self.count[unique_values] += counts
        
        
        # vq_loss = self.ContrastLoss(distances, k=5)
        
        # # # EMA update
        if mode=='train':
            self.EMAUpdate(h_flat, distances, self.ema_k, self.ema_lambda)
        
        vq_loss = self.MSE(h.detach(), quantized) + 0.25*self.MSE(h, quantized.detach())
        if attn_mask is not None:
            vq_loss = ((vq_loss*attn_mask[:,:,None]).sum(dim=1)/attn_mask[:,:,None].sum(dim=1)).mean()
        else:
            vq_loss = vq_loss.mean()
        
        quantized = self.proj_inv(quantized)
        return quantized, code, vq_loss


class SoftVQLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, vq_dim, ema_k=5, ema_lambda=0.95, ema_weight=100):
        super(SoftVQLayer, self).__init__()
        self.init = True
        self.ema_k = ema_k
        self.ema_lambda = ema_lambda
        self.ema_weight = ema_weight
        self.num_embeddings = num_embeddings
        self.vq_dim = vq_dim
        self.proj = nn.Linear(embedding_dim, vq_dim)
        self.proj_inv = nn.Linear(vq_dim, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, vq_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.register_buffer('count', torch.zeros(num_embeddings))
        self.init=False
        self.MSE = nn.HuberLoss(reduction='none')
        
    def project(self, h): 
        h = self.proj(h)
        h = h/h.norm(dim=-1, keepdim=True)
        return h
    
    def project_inv(self, h):
        h = self.proj_inv(h)
        return h

    def embed_id(self, vq_id):
        self.embedding.weight.data = self.embedding.weight.data/self.embedding.weight.data.norm(dim=-1, keepdim=True)
        return self.proj_inv(self.embedding(vq_id))

    def attention(self, H, C, temperature=1):
        distances = torch.sum(H ** 2, dim=1, keepdim=True) \
                  + torch.sum(C ** 2, dim=1) \
                  - 2 * torch.matmul(H, C.t())
        A = F.softmax(-distances/temperature, dim=1)
        return A
    
    def get_vq(self,h,attn_mask,temperature):
        h = self.proj(h)
        h = h/torch.norm(h, dim=-1, keepdim=True)
        h_flat = h[attn_mask==1]
        self.embedding.weight.data = self.embedding.weight.data/self.embedding.weight.data.norm(dim=-1, keepdim=True)
        A = self.attention(h_flat, self.embedding.weight, temperature)
        code = A.argmax(dim=-1)
        h_vq = self.embedding(code)
        
        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        vq_code = torch.zeros(h.shape[:2], device = h.device, dtype = torch.long)
        vq_code[attn_mask==1] = code
        quantized = self.proj_inv(quantized)
        return vq_code, quantized
        
    
    def forward(self, h, attn_mask=None, mode='train', temperature=1):
        h = self.proj(h)
        h = h/torch.norm(h, dim=-1, keepdim=True)
        if attn_mask is None:
            attn_mask = torch.ones_like(h[:,:,0])
        h_flat = h[attn_mask==1]
        
        self.embedding.weight.data = self.embedding.weight.data/self.embedding.weight.data.norm(dim=-1, keepdim=True)

        
        A = self.attention(h_flat, self.embedding.weight, temperature)
        code = A.argmax(dim=-1)
        
        if mode=='train':
            h_vq = A@self.embedding.weight
        else:
            h_vq = self.embedding(code)
        
        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        vq_loss = 0
        
        vq_code = torch.zeros(h.shape[:2], device = h.device, dtype = torch.long)
        vq_code[attn_mask==1] = code

        quantized = self.proj_inv(quantized)
        return quantized, vq_code, vq_loss


class SoftGVQLayer(nn.Module):
    def __init__(self, log2_num_embeddings, embedding_dim, vq_dim):
        super(SoftGVQLayer, self).__init__()
        self.init = True
        self.log2_num_embeddings = log2_num_embeddings
        self.vq_dim = vq_dim
        self.proj = nn.Linear(embedding_dim, vq_dim)
        self.proj_inv = nn.Linear(vq_dim, embedding_dim)
        self.embedding = nn.Embedding(int(2*log2_num_embeddings), vq_dim)
        # print()
        # self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.init=False
        self.MSE = nn.HuberLoss(reduction='none')
        
    def project(self, h): 
        h = self.proj(h)
        h = h/h.norm(dim=-1, keepdim=True)
        return h
    
    def project_inv(self, h):
        h = self.proj_inv(h)
        return h

    def embed_id(self, vq_id):
        self.embedding.weight.data = self.embedding.weight.data/self.embedding.weight.data.norm(dim=-1, keepdim=True)
        shift = 2*torch.range(0,self.log2_num_embeddings-1, device=vq_id.device, dtype=torch.long)
        vq_id = vq_id[:, :, :-1]+shift
        quantized = self.proj_inv(self.embedding(vq_id).sum(-2))
        return quantized

    def attention(self, H, C, temperature=1):
        distances = torch.sum(H ** 2, dim=1, keepdim=True) \
                  + torch.sum(C ** 2, dim=1) \
                  - 2 * torch.matmul(H, C.t())
        N, M = distances.shape
        distances = distances.reshape(N, M//2, 2)
        A = F.softmax(-distances/temperature, dim=-1)
        A = A.reshape(N,M)
        return A
    
    def get_vq(self,h,attn_mask,temperature, proj_inv=True):
        h = self.proj(h)
        h = h/(torch.norm(h, dim=-1, keepdim=True)+1e-8)
        h_flat = h[attn_mask==1]
        self.embedding.weight.data = self.embedding.weight.data/self.embedding.weight.data.norm(dim=-1, keepdim=True)
        A = self.attention(h_flat, self.embedding.weight, temperature)
        code = A.argmax(dim=-1)
        code_vq = A.view(h_flat.shape[0], -1, 2).argmax(dim=-1)
        h_vq = self.embedding(code)
        
        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        vq_code = torch.zeros(h.shape[:2] + (self.log2_num_embeddings,), device = h.device, dtype = torch.long)
        vq_code[attn_mask==1] = code_vq
        if proj_inv:
            quantized = self.proj_inv(quantized)
        return vq_code, quantized
        
    
    def forward(self, h, attn_mask=None, mode='train', temperature=1, proj_inv=True):
        h = self.proj(h)
        h = h/(torch.norm(h, dim=-1, keepdim=True)+1e-8)
        if attn_mask is None:
            attn_mask = torch.ones_like(h[:,:,0])
        h_flat = h[attn_mask==1]
        self.embedding.weight.data = self.embedding.weight.data/(self.embedding.weight.data.norm(dim=-1, keepdim=True)+1e-8)
        
        A = self.attention(h_flat, self.embedding.weight, temperature)
        N,M = A.shape
        code = A.reshape(N,M//2,2).argmax(dim=-1)
        
        if mode=='train':
            h_vq = A@self.embedding.weight
        else:
            shift = 2*torch.range(0,self.log2_num_embeddings-1, device=A.device, dtype=torch.long)
            index = code+shift[None,:]
            h_vq = self.embedding(index).sum(dim=1)
        
        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        vq_loss = 0
        
        vq_code = torch.zeros(h.shape[:2], device = h.device, dtype = torch.long)
        shift = 2**torch.range(0,self.log2_num_embeddings-1, device=A.device, dtype=torch.long)
        vq_code[attn_mask==1] = (code*shift[None,:]).sum(dim=-1)
        if proj_inv:
            quantized = self.proj_inv(quantized)
        return quantized, vq_code, vq_loss
        


class LookupFreeQuantizer(nn.Module):
    def __init__(self, vocab_size: int=None, hidden_size=480):
        super(LookupFreeQuantizer, self).__init__()
        self.proj = nn.Linear(hidden_size, int(math.log2(vocab_size)))
        self.vocab_size = vocab_size
        self.proj_inv = nn.Linear(int(math.log2(vocab_size)), hidden_size)
        self.MSE = nn.L1Loss(reduction='none')

    def sign(self, z: torch.Tensor):
        q_z = torch.sign(z)
        q_z[q_z == 0] = 1
        return q_z

    def token_index(self, q_z: torch.Tensor):
        indices = (torch.arange(q_z.size(-1), dtype=torch.float32)).to(q_z.device)
        tokens = torch.sum(2**indices * (q_z > 0).float(), dim=-1)
        return tokens
    
    def get_vq(self, z: torch.Tensor, attn_mask=None, mode='train', temperature=1.0):
        z = F.tanh(self.proj(z))
        if self.vocab_size is not None:
            assert z.size(-1)==math.log2(self.vocab_size)

        q_z = self.sign(z)
        q_z = z + (q_z - z).detach()
      #  index = self.token_index(q_z)
        index = (q_z > 0).long()

        quantized = self.proj_inv(q_z)
        return index, quantized

    def embed_id(self, q_z):
        vec = dec2bin(q_z, int(math.log2(self.vocab_size)))
        vec = vec.flip(dims=[-1])
        vec = self.sign(vec-0.5)
        quantized = self.proj_inv(vec)
        return quantized

    def forward(self, z: torch.Tensor, attn_mask=None, mode='train', temperature=1.0):
        z = F.tanh(self.proj(z))
        if self.vocab_size is not None:
            assert z.size(-1)==math.log2(self.vocab_size)

        q_z = self.sign(z)
        q_z = z + (q_z - z).detach()
        # if mode == 'train':
        #     q_z = torch.sigmoid(z/temperature)
        # else:
        #     # q_z = self.sign(z)
        #     temperature = 1e-5
        #     q_z = torch.sigmoid(z/temperature)
        vq_loss = self.MSE(z.detach(), q_z) + 0.25*self.MSE(z, q_z.detach())
        if attn_mask is not None:
            vq_loss = ((vq_loss*attn_mask[:,:,None]).sum(dim=1)/attn_mask[:,:,None].sum(dim=1)).mean()
        else:
            vq_loss = vq_loss.mean()
        
        index = self.token_index(q_z)
        q_z = self.proj_inv(q_z)
        return q_z, index.long(), vq_loss


class SoftCVQLayer(nn.Module):
    def __init__(self, log2_num_embeddings, embedding_dim, vq_dim, condition_layer=6, sphere=True):
        super(SoftCVQLayer, self).__init__()
        self.init = True
        self.log2_num_embeddings = log2_num_embeddings
        # 生成从0到65535的整数范围
        int_range = torch.arange(0, 2**log2_num_embeddings)
        bool_vectors = (int_range[:, None] & (1 << torch.arange(log2_num_embeddings-1, -1, -1))) > 0

        self.register_buffer('embedding', bool_vectors.float())
        self.sphere = sphere

        '''
        SoftBV-vq16-conditional: 使用两层MLP将bool vector映射到log2_num_embeddingslog2_num_embeddings相同维度, 不归一化
        SoftBV-vq16-conditional-sphere: 使用两层MLP将bool vector映射到log2_num_embeddings相同维度, 归一化
        SoftBV-vq16-conditional-mlp2-vqdim32: 使用两层MLP将bool vector映射到vq_dim, 不归一化
        SoftBV-vq16-conditional-sphere-vqdim32: 使用两层MLP将bool vector映射到vq_dim, 归一化
        SoftBV-vq16-conditional-mlp3-vqdim32: 使用三层MLP将bool vector映射到vq_dim, 不归一化
        '''
        hidden_dim = 1024

        if condition_layer <=3:
            layers = [nn.Linear(log2_num_embeddings, hidden_dim), nn.ReLU()]
            for _ in range(condition_layer - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim)),
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, vq_dim))
            self.embedding_mlp = nn.Sequential(*layers)
        else:
            layers = [nn.Linear(log2_num_embeddings, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
            for _ in range(condition_layer - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, vq_dim))
            self.embedding_mlp = nn.Sequential(*layers)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True)
        # self.proj_trans =  nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.proj = nn.Linear(embedding_dim, vq_dim)
        self.proj_inv = nn.Linear(vq_dim, embedding_dim)
        
        self.init=False
        self.MSE = nn.HuberLoss(reduction='none')
        
    def project(self, h): 
        h = self.proj(h)
        return h
    
    def project_inv(self, h):
        h = self.proj_inv(h)
        return h

    def embed_id(self, vq_id, level=None):
        embed = self.embedding_mlp(self.embedding)
        if self.sphere:
            embed = embed/torch.norm(embed, dim=-1, keepdim=True) # spherical
        return self.proj_inv(embed[vq_id])
    
    def get_code(self,h, attn_mask = None, temperature = 1e-5):
        h = self.proj(h)

        embed = self.embedding_mlp(self.embedding)
        if self.sphere:
            h = self.normalize(h) # spherical
            embed = self.normalize(embed) # spherical
        
        if attn_mask is None:
            attn_mask = torch.ones_like(h[:,0])

        h_flat = h[attn_mask==1]
        A, _ = self.attention(h_flat, embed, temperature)
        vq_code = A.argmax(dim=-1)
        return vq_code
    
    def decimal2binary(self, vqids):
        return self.embedding[vqids]
    
    def binary2decimal(self, binary_vector):
        base = 2 ** torch.arange(binary_vector.size(-1) - 1, -1, -1, device=binary_vector.device)
        vqids = (binary_vector * base).long().sum(dim=-1)
        return vqids

    def attention(self, H, C, temperature=1):
        ### ============= 这里有个离谱的bug,是pytorch或者硬件的问题, 见notion
        ### 在batch size增大后, argmax与argmin似乎不再返回最小索引,当两个val数值一样时
        # distances = torch.sum(H ** 2, dim=1, keepdim=True) \
        #           + torch.sum(C ** 2, dim=1) \
        #           - 2 * torch.matmul(H, C.t())

        # H_high_precision = H.to(torch.float64)
        # C_high_precision = C.to(torch.float64)

        alpha = 1/temperature
        # stable attention
        distances = - 2 * (alpha-1)* (H@C.t()).detach() - 2 * H@C.t()


        # distances = StableAttn.apply(H, C, temperature)

        
        # distances = torch.cdist(H, C, compute_mode='use_mm_for_euclid_dist')
        A = F.softmax(-distances, dim=1)
        # A = gumbel_softmax(-distances, temperature)
        # A = F.softmax(-distances, dim=1)

        # min_val = distances.min(dim=-1)[0]
        # vq_code = (distances<min_val[:,None]+5e-7).float().argmax(dim=-1)
        vq_code = distances.argmin(dim=-1)
        return A,  vq_code

    def normalize(self, x):
        return x/(torch.norm(x, dim=-1, keepdim=True)+1e-6)
    
    def get_vq(self,h, attn_mask = None, temperature = 1e-5):
        h = self.proj(h)

        embed = self.embedding_mlp(self.embedding)
        if self.sphere:
            h = self.normalize(h) # spherical
            embed = self.normalize(embed) # spherical
        
        if attn_mask is None:
            attn_mask = torch.ones_like(h[:,0])

        h_flat = h[attn_mask==1]
        A, code = self.attention(h_flat, embed, temperature)
        # code = A.argmax(dim=-1)
        h_vq = embed[code]
        
        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        vq_code = torch.zeros(h.shape[:2], device = h.device, dtype = torch.long)
        vq_code[attn_mask==1] = code
        quantized = self.proj_inv(quantized)
        return vq_code, quantized
    
    def entropy_loss(self, P, Q):
        return -torch.sum(P * torch.log(Q))


    def forward(self, h_in, attn_mask=None, mode='train', temperature=1, vqshortcut=False, frozen=False):
        # h_in = self.proj_trans(h_in)
        h = self.proj(h_in)

        embed = self.embedding_mlp(self.embedding)

        if self.sphere:
            h = self.normalize(h) # spherical
            embed = self.normalize(embed) # spherical

        if attn_mask is None:
            attn_mask = torch.ones_like(h[:,0])
        h_flat = h[attn_mask==1]
        
        A, code = self.attention(h_flat, embed, temperature)
        # A, code = self.attention(h_flat, embed, 1.0)


        mat = (embed@embed.permute(1,0))
        indices = torch.arange(mat.size(0))
        mat[indices, indices] = -1
        vq_loss = mat.max(dim=-1)[0].mean()

        # entropy = -torch.sum(A * torch.log(A), dim=-1)
        # entropy_neg = -(entropy*attn_mask).sum()/attn_mask.sum()

        # 直接使用cosine similarity最稳定, softmax之后可能有浮点误差，在取argmax的时候让index找错,从而导致模型表现异常
        # code = flat_affinity.argmax(dim=-1) 
        code = torch.multinomial(A, 1).view(-1)
        code_one_hot = F.one_hot(code, num_classes=A.size(-1))
        c = code_one_hot*(1-A) + (1-code_one_hot)*(-A)
        A2 = A + c.detach()
        h_vq = h_flat + A2@embed - h_flat.detach()  ## newvq2
        # h_vq = h_vq@embed

        # if mode=='train':
        #     h_vq = A@embed
        # else:
        #     h_vq = embed[code]
        
        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        # vq_loss = torch.zeros(1, device=h.device)
        
        vq_code = torch.zeros(h.shape[:-1], device = h.device, dtype = torch.long)
        vq_code[attn_mask==1] = code

        quantized = self.proj_inv(quantized)

        if vqshortcut and temperature>0.02:
            N = h_in.shape[0]
            keep_idx = torch.randperm(N)[:int(1.0*temperature*N)]
            replace = torch.zeros(h_in.shape[0],1,device=h_in.device,dtype=h_in.dtype)
            replace[keep_idx] = 1
            quantized = quantized*(1-replace) + h_in*replace
        


        # # 计算每个随机变量的平均值
        # probs = F.softmax(flat_affinity, dim=-1)
        # log_probs = F.log_softmax(flat_affinity + 1e-8, dim=-1)
        # mean_probs = torch.mean(A, dim=0)
        # # 计算均匀分布的目标
        # uniform_target = torch.full_like(mean_probs, 1.0 / mean_probs.size(-1))
        
        # # 计算每个随机变量的熵
        # sample_entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # # 计算所有样本的平均熵
        # avg_sample_entropy = torch.mean(sample_entropy)
        
        # # 计算均匀分布的熵
        # uniform_entropy = -torch.sum(uniform_target * torch.log(uniform_target + 1e-8))
        
        # # 计算目标：使得平均熵接近于均匀分布的熵
        # vq_loss = -0.1*(avg_sample_entropy - uniform_entropy)
        # vq_loss = 0

        
        return quantized, vq_code, vq_loss


class HierCVQLayer(nn.Module):
    def __init__(self, log2_num_embeddings, embedding_dim, vq_dim, levels, condition_layer=6, sphere=True):
        super(HierCVQLayer, self).__init__()
        self.init = True
        self.log2_num_embeddings = log2_num_embeddings
        self.levels = levels
        # self.register_buffer('embedding', bool_vectors.float())
        self.sphere = sphere
        hidden_dim = 1024

        # if condition_layer <=3:
        #     layers = [nn.Linear(log2_num_embeddings, hidden_dim), nn.ReLU()]
        #     for _ in range(condition_layer - 2):
        #         layers.append(nn.Linear(hidden_dim, hidden_dim)),
        #         layers.append(nn.ReLU())
        #     layers.append(nn.Linear(hidden_dim, vq_dim))
        #     self.embedding_mlp = nn.Sequential(*layers)
        # else:
        #     layers = [nn.Linear(log2_num_embeddings, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        #     for _ in range(condition_layer - 2):
        #         layers.append(nn.Linear(hidden_dim, hidden_dim))
        #         layers.append(nn.BatchNorm1d(hidden_dim))
        #         layers.append(nn.ReLU())
        #     layers.append(nn.Linear(hidden_dim, vq_dim))
        #     self.embedding_mlp = nn.Sequential(*layers)

        self.embedding_mlp = nn.Sequential(*[self.build_condition_layer(log2_num_embeddings,hidden_dim,condition_layer,vq_dim) for i in range(log2_num_embeddings)])
        

        self.proj = nn.Linear(embedding_dim, vq_dim)
        self.proj_inv = nn.Linear(vq_dim, embedding_dim)
        
        self.init=False
        self.MSE = nn.HuberLoss(reduction='none')

    def build_condition_layer(self,log2_num_embeddings,hidden_dim,condition_layer,vq_dim):
        layers = [nn.Linear(log2_num_embeddings, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        for _ in range(condition_layer - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, vq_dim))
        return nn.Sequential(*layers)

    def get_bvectors(self, num_embeddings):
        int_range = torch.arange(0, 2**num_embeddings)
        bool_vectors = (int_range[:, None] & (1 << torch.arange(num_embeddings-1, -1, -1))) > 0
        return bool_vectors.float()
        
    def project(self, h): 
        h = self.proj(h)
        return h
    
    def project_inv(self, h):
        h = self.proj_inv(h)
        return h

    def embed_id(self, vq_id, level):
        bvectors = self.get_bvectors(level)
        bvectors = F.pad(bvectors, (self.log2_num_embeddings-level,0,0,0), value=-1).to(vq_id.device)
        bvectors = bvectors.to(self.embedding_mlp[level-1][0].weight.dtype)

        embed = self.embedding_mlp[level-1](bvectors)
        
        if self.sphere:
            embed = embed/torch.norm(embed, dim=-1, keepdim=True) # spherical
        return self.proj_inv(embed[vq_id])
    
    def get_code(self,h, attn_mask = None, temperature = 1e-5):
        h = self.proj(h)

        embed = self.embedding_mlp(self.embedding)
        if self.sphere:
            h = self.normalize(h) # spherical
            embed = self.normalize(embed) # spherical
        
        if attn_mask is None:
            attn_mask = torch.ones_like(h[:,0])

        h_flat = h[attn_mask==1]
        A, _ = self.attention(h_flat, embed, temperature)
        vq_code = A.argmax(dim=-1)
        return vq_code
    
    def decimal2binary(self, vqids):
        return self.embedding[vqids]
    
    def binary2decimal(self, binary_vector):
        base = 2 ** torch.arange(binary_vector.size(-1) - 1, -1, -1, device=binary_vector.device)
        vqids = (binary_vector * base).long().sum(dim=-1)
        return vqids

    def attention(self, H, C, temperature=1):
        alpha = 1/temperature
        distances = - 2 * (alpha-1)* (H@C.t()).detach() - 2 * H@C.t()

        A = F.softmax(-distances, dim=1)
        vq_code = distances.argmin(dim=-1)
        return A,  vq_code

    def normalize(self, x):
        return x/(torch.norm(x, dim=-1, keepdim=True)+1e-6)
    
    def get_vq(self,h, attn_mask = None, temperature = 1e-5):
        h = self.proj(h)

        embed = self.embedding_mlp(self.embedding)
        if self.sphere:
            h = self.normalize(h) # spherical
            embed = self.normalize(embed) # spherical
        
        if attn_mask is None:
            attn_mask = torch.ones_like(h[:,0])

        h_flat = h[attn_mask==1]
        A, code = self.attention(h_flat, embed, temperature)
        h_vq = embed[code]
        
        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        vq_code = torch.zeros(h.shape[:2], device = h.device, dtype = torch.long)
        vq_code[attn_mask==1] = code
        quantized = self.proj_inv(quantized)
        return vq_code, quantized
    
    def entropy_loss(self, P, Q):
        return -torch.sum(P * torch.log(Q))

    def mask_code(self, code):
        code_b = self.decimal2binary(code)
        index_tensor = torch.randint(0, self.log2_num_embeddings, (code.shape[0],), device=code.device)

        # index_tensor = torch.ones_like(index_tensor)*8
        def create_mask(index_tensor, K):
            N = len(index_tensor)
            mask = torch.zeros((N, K), dtype=torch.bool, device=index_tensor.device)
            for i, idx in enumerate(index_tensor):
                mask[i, :idx] = True
            return mask
        mask = create_mask(index_tensor, self.log2_num_embeddings)
        code_b = code_b*(~mask)
        
        code = self.binary2decimal(code_b)
        return code

# torch.save(embed, f'/huyuqi/xmyu/FoldToken2/FoldToken4/results/FT4_PDB_level5_to_12_fixdata_continue_lenfree/vqvec_level{level}.pt')

# device = 'cuda'
# FT3 = torch.load('/huyuqi/xmyu/FoldToken2/FoldToken4/results/FT4_PDB_level5_to_12_fixdata_continue_lenfree/vqvec_level6.pt').to(device)
# FT3 = FT3/(torch.norm(FT3, dim=-1, keepdim=True)+1e-6)
# mat = FT3@FT3.permute(1,0)
# mat = mat - 10*torch.eye(mat.shape[0], device=device)
# FT3_val, FT3_idx = mat.max(dim=-1)
# print(FT3_val.mean(), FT3_val.min(), FT3_val.max())

    def sample_code(self, h, level, temperature, attn_mask):
        bvectors = self.get_bvectors(level)
        bvectors = F.pad(bvectors, (self.log2_num_embeddings-level,0,0,0), value=-1).to(h.device)

        embed = self.embedding_mlp[level-1](bvectors)
        # embed = self.embedding_mlp[0](bvectors)

        if self.sphere:
            h = self.normalize(h) # spherical
            embed = self.normalize(embed) # spherical

        if attn_mask is None:
            attn_mask = torch.ones_like(h[:,0])
        h_flat = h[attn_mask==1]
        
        A, _ = self.attention(h_flat, embed, temperature)

        mat = (embed@embed.permute(1,0))
        indices = torch.arange(mat.size(0))
        mat[indices, indices] = -1
        vq_loss = mat.max(dim=-1)[0].mean()

        
        code = torch.multinomial(A, 1).view(-1)
        code_one_hot = F.one_hot(code, num_classes=A.size(-1))
        c = code_one_hot*(1-A) + (1-code_one_hot)*(-A)
        A2 = A + c.detach()
        h_vq = h_flat + A2@embed - h_flat.detach()
        

        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        return quantized, code, vq_loss


    def forward(self, h_in, attn_mask=None, mode='train', temperature=1, vqshortcut=False, level = 8):
        h = self.proj(h_in)
        if h.isnan().any():
            h = torch.rand_like(h)
            # print(h)

        if mode == 'train':
            quantized = 0
            vq_loss = 0
            p = torch.rand(h.shape[0], device = h.device)
            for idx, level in enumerate(self.levels):
                quantized_, code, vq_loss_ = self.sample_code(h, level, temperature, attn_mask)
                select = (idx/len(self.levels)<p) & (p<(idx+1)/len(self.levels))
                quantized = quantized + quantized_*select[:,None]
                vq_loss += vq_loss_

            # quantized = quantized[0]*p[:,None] + quantized[1]*(1-p)[:,None]
            vq_loss = vq_loss/len(self.levels)
        else:
            quantized, code, vq_loss = self.sample_code(h, level, temperature, attn_mask)

        # quantized, code, vq_loss = self.sample_code(h, level, temperature, attn_mask)

        # ## ================== hierachy
        # quantized, code, vq_loss = self.sample_code(h, 12, temperature, attn_mask)
        # level = 8
        # map = torch.load(f'/huyuqi/xmyu/FoldToken2/FoldToken4/results/FT4_PDB_level5_to_12_fixdata_continue_lenfree/map12to{level}.pt')
        
        # bvectors = self.get_bvectors(level)
        # bvectors = F.pad(bvectors, (self.log2_num_embeddings-level,0,0,0), value=-1).to(h.device)

        # embed = self.embedding_mlp[level-1](bvectors)
        # embed = self.normalize(embed) 
        # print(map[code])
        # quantized = embed[map[code]]
        # ## ================== 

        vq_code = code
        # vq_code = torch.zeros(h.shape[:-1], device = h.device, dtype = torch.long)
        # vq_code[attn_mask==1] = code

        quantized = self.proj_inv(quantized)

        if vqshortcut:
            # shortcur_prob = 0.3
            N = h_in.shape[0]
            keep_idx = torch.randperm(N)[:int(1.0*temperature*N)]
            replace = torch.zeros(h_in.shape[0],1,device=h_in.device,dtype=h_in.dtype)
            replace[keep_idx] = 1
            quantized = quantized*(1-replace) + h_in*replace
        


        
        return quantized, vq_code, vq_loss


def round_ste(z):
    """Round with straight through gradients."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()



class FSQ(nn.Module):
    """Quantizer."""

    def __init__(self, levels: list[int], embedding_dim, vq_dim, eps: float = 1e-3):
        super(FSQ, self).__init__()
        self._levels = levels
        self._eps = eps
        self._levels_np = np.asarray(levels)
        self._basis = np.concatenate(([1], np.cumprod(self._levels_np[:-1]))).astype(np.int64)  # 修改此处为 np.int64
        self._implicit_codebook = self.indexes_to_codes(torch.arange(self.codebook_size, dtype=torch.int64))
        self.proj = nn.Linear(embedding_dim, vq_dim)
        self.proj_inv = nn.Linear(vq_dim, embedding_dim)

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return np.prod(self._levels)

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_np - 1) * (1 - self._eps) / 2
        offset = np.where(self._levels_np % 2 == 1, 0.0, 0.5)
        shift = np.tan(offset / half_l)
        return torch.tanh(z + torch.tensor(shift, dtype=z.dtype, device=z.device)) * torch.tensor(half_l, dtype=z.dtype, device=z.device) - torch.tensor(offset, dtype=z.dtype, device=z.device)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))

        # Renormalize to [-1, 1].
        half_width = self._levels_np // 2
        return quantized / torch.tensor(half_width, dtype=z.dtype, device=z.device)

    def _scale_and_shift(self, zhat_normalized):
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_np // 2
        return (zhat_normalized * torch.tensor(half_width, dtype=zhat_normalized.dtype, device=zhat_normalized.device)) + torch.tensor(half_width, dtype=zhat_normalized.dtype, device=zhat_normalized.device)

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_np // 2
        return (zhat - torch.tensor(half_width, dtype=zhat.dtype, device=zhat.device)) / torch.tensor(half_width, dtype=zhat.dtype, device=zhat.device)

    def codes_to_indexes(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * torch.tensor(self._basis, dtype=zhat.dtype, device=zhat.device)).sum(dim=-1).type(torch.int64)  # 修改此处为 torch.int64

    def indexes_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of `codes_to_indexes`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = np.mod(
            np.floor_divide(indices.cpu().numpy(), self._basis), self._levels_np
        )
        return self._scale_and_shift_inverse(torch.tensor(codes_non_centered, dtype=indices.dtype, device=indices.device))
    
    def forward(self, h_in, temperature=0, mode='train'):
        h = self.proj(h_in)
        quantized = self.quantize(h)
        vq_code = self.codes_to_indexes(quantized)
        quantized = self.proj_inv(quantized)
        return quantized, vq_code, 0