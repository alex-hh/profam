import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import math
from src.tools.geo_utils import compute_frenet_frames
from src.tools.affine_utils import Rigid
from openfold.data import residue_constants, protein
from transformers import AutoTokenizer
from openfold.utils import data_utils as du
import numpy as np
from openfold.data import data_transforms
import random
from openfold.utils import rigid_utils

class CropPadData:
    def __init__(self, pad, tokenizer):
        self.pad = pad
        self.tokenizer = tokenizer
    
    def __call__(self, data):
        seqs, angles, coords = data['seqs'], data['angles'], data['coords']
        pad = self.pad
        ESM_tokenizer = self.tokenizer
        
        ## ================== cropping
        if angles.shape[0]>pad:
            start_idx = torch.randint(size=(1,),low=0, high=angles.shape[0] - pad)
            end_idx = start_idx + pad
            assert end_idx < angles.shape[0]
            angles = angles[start_idx:end_idx]
            seqs = seqs[start_idx:end_idx]
            coords = coords[start_idx:end_idx]
            assert angles.shape[0] == angles.shape[0]

        # Create attention mask. 0 indicates masked
        l = min(pad, angles.shape[0])
        attn_mask = torch.zeros(size=(pad,))
        attn_mask[:l] = 1.0

        # Perform padding
        if angles.shape[0] < pad:
            angles = F.pad(
                angles,
                (0, 0, 0, pad - angles.shape[0])
            )

            seqs = F.pad(
                seqs,
                (0,0,0, pad - seqs.shape[0]),
                mode="constant",
                value=ESM_tokenizer.pad_token_id,
            )
            
            coords = F.pad(
                coords,
                (0,0,0, pad - coords.shape[0]),
            )


        # Create position IDs
        position_ids = torch.arange(start=0, end=pad, step=1, dtype=torch.long)
        data.update({"seqs":seqs,
                     "angles":angles,
                     "coords":coords,
                     "attn_mask":attn_mask,
                     "position_ids":position_ids})
        return data
    
class FrameData:
    def __init__(self):
        pass
    
    def __call__(self, data):
        coords, mask = data['coords'], data['attn_mask']
        coords, mask = coords[None,...], mask[None,...]
        trans = coords - torch.mean(coords, dim=1, keepdim=True) # 所有坐标中心化
        rots = compute_frenet_frames(trans, mask)
        frame = Rigid(rots.to(torch.bfloat16), trans.to(torch.bfloat16))
        data.update({"frame":frame[0]})
        return data
    
class DataTrasform:
    def __init__(self) -> None:
        pass

    def Atom37(self, aatype, all_atom_positions, all_atom_mask, chain_idx, res_idx, res_mask, max_res_num=256, is_training=True):
        chain_feats = {"aatype":aatype,
                       "all_atom_positions": all_atom_positions,
                       "all_atom_mask":all_atom_mask}
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)


        # Re-number residue indices for each chain such that it starts from 1.
        # Randomize chain indices.
        new_res_idx = np.zeros_like(res_idx)
        new_chain_idx = np.zeros_like(res_idx)
        all_chain_idx = np.unique(chain_idx).tolist()
        shuffled_chain_idx = np.array(
            random.sample(all_chain_idx, len(all_chain_idx))) - np.min(all_chain_idx) + 1
        for i,chain_id in enumerate(all_chain_idx):
            chain_mask = (chain_idx == chain_id).astype(int)
            chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
            new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

            # Shuffle chain_index
            replacement_chain_id = shuffled_chain_idx[i]
            new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

        # To speed up processing, only take necessary features
        chain_feats = {
            'aatype': chain_feats['aatype'],
            'seq_idx': new_res_idx,
            'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
            'residue_index': res_idx,
            'res_mask': res_mask,
            'atom37_pos': chain_feats['all_atom_positions'],
            'atom37_mask': chain_feats['all_atom_mask'],
            'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'],
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'],
        }

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(
            chain_feats['rigidgroups_0'])[:, 0]
        diffused_mask = np.ones_like(chain_feats['res_mask'])
        if np.sum(diffused_mask) < 1:
            raise ValueError('Must be diffused')
        fixed_mask = 1 - diffused_mask
        chain_feats['fixed_mask'] = fixed_mask
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        chain_feats['sc_ca_t'] = torch.zeros_like(gt_bb_rigid.get_trans())

        final_feats = du.pad_feats(chain_feats, max_res_num)
        

        return final_feats

class Chain2AF2:
    def __init__(self):
        pass

    def __call__(self, data):
        '''
        seqs: [L]
        coords: [L,3]
        '''
        seqs, angles, coords, attn_mask = data['seqs'], data['angles'], data['coords'], data['attn_mask']
        chain_id='A'
        b_factors=None

        chain_id = du.chain_str_to_int(chain_id)
        N = len(seqs)
        atom_positions = np.zeros((N, 37, 3))
        atom_positions[:,residue_constants.atom_order['CA']] = coords

        atom_mask = np.zeros((N, 37))
        atom_mask[:,residue_constants.atom_order['CA']] = attn_mask

        aatype, residue_index, chain_ids = [], [], []
        for idx, res_shortname in enumerate(seqs):
            residue_index.append(idx)
            chain_ids.append(chain_id)
        aatype = seqs.view(-1).cpu().numpy()
        residue_index = np.array(residue_index)
        chain_ids = np.array(chain_ids)

        if b_factors is None:
            b_factors = [100 for i in range(N)]

        AF2Data = protein.Protein(
            atom_positions=atom_positions, # [121,37,3]
            atom_mask=atom_mask, # [121,37]
            aatype=aatype, # [121]
            residue_index=residue_index, # [121]
            chain_index=chain_ids, # [121]
            b_factors=np.array(b_factors)) # [121,37]
        
        transform = DataTrasform()

        data = transform.Atom37(AF2Data.aatype, AF2Data.atom_positions, AF2Data.atom_mask, AF2Data.chain_index, AF2Data.residue_index, AF2Data.atom_mask, max_res_num=seqs.shape[0])
        return {'FrameData':data}


class MLM:
    """
    ## Masked LM (MLM)

    This class implements the masking procedure for a given batch of token sequences.
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/mlm/__init__.py
    """
    def __init__(self, *,
                 padding_token: int=1, mask_token: int=32, no_mask_tokens: List[int] = [], n_tokens: int=33,
                 masking_prob: float = 0.15, randomize_prob: float = 0.1, no_change_prob: float = 0.1,
                 ):
        """
        * `padding_token` is the padding token `[PAD]`.
          We will use this to mark the labels that shouldn't be used for loss calculation.
        * `mask_token` is the masking token `[MASK]`.
        * `no_mask_tokens` is a list of tokens that should not be masked.
        This is useful if we are training the MLM with another task like classification at the same time,
        and we have tokens such as `[CLS]` that shouldn't be masked.
        * `n_tokens` total number of tokens (used for generating random tokens)
        * `masking_prob` is the masking probability
        * `randomize_prob` is the probability of replacing with a random token
        * `no_change_prob` is the probability of replacing with original token
        """
        self.n_tokens = n_tokens
        self.no_change_prob = no_change_prob
        self.randomize_prob = randomize_prob
        self.masking_prob = masking_prob
        self.no_mask_tokens = no_mask_tokens + [padding_token, mask_token]
        self.padding_token = padding_token
        self.mask_token = mask_token

    def __call__(self, x: torch.Tensor):
        """
        * `x` is the batch of input token sequences.
         It's a tensor of type `long` with shape `[batch_size, seq_len]`.
        """
        # Mask `masking_prob` of tokens
        full_mask = torch.rand(x.shape, device=x.device) < self.masking_prob
        # Unmask `no_mask_tokens`
        for t in self.no_mask_tokens:
            full_mask &= x != t

        # A mask for tokens to be replaced with original tokens
        unchanged = full_mask & (torch.rand(x.shape, device=x.device) < self.no_change_prob)
        # A mask for tokens to be replaced with a random token
        random_token_mask = full_mask & (torch.rand(x.shape, device=x.device) < self.randomize_prob)
        # Indexes of tokens to be replaced with random tokens
        random_token_idx = torch.nonzero(random_token_mask, as_tuple=True)
        # Random tokens for each of the locations
        random_tokens = torch.randint(0, self.n_tokens, (len(random_token_idx[0]),), device=x.device)
        # The final set of tokens that are going to be replaced by `[MASK]`
        mask = full_mask & ~random_token_mask & ~unchanged

        # Make a clone of the input for the labels
        y = x.clone()

        # Replace with `[MASK]` tokens;
        # note that this doesn't include the tokens that will have the original token unchanged and
        # those that get replace with a random token.
        x.masked_fill_(mask, self.mask_token)
        # Assign random tokens
        x[random_token_idx] = random_tokens

        # Assign token `[PAD]` to all the other locations in the labels.
        # The labels equal to `[PAD]` will not be used in the loss.
        y.masked_fill_(~full_mask, self.padding_token)

        return x, y, full_mask
    
    
class MLM_Angle:
    """
    ## Masked LM (MLM)

    This class implements the masking procedure for a given batch of token sequences.
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/mlm/__init__.py
    """
    def __init__(self, *,
                 padding_token: int=1, mask_token: int=-1, no_mask_tokens: List[int] = [], n_tokens: int=33,
                 masking_prob: float = 0.15, randomize_prob: float = 0.1, no_change_prob: float = 0.1,
                 ):
        """
        * `padding_token` is the padding token `[PAD]`.
          We will use this to mark the labels that shouldn't be used for loss calculation.
        * `mask_token` is the masking token `[MASK]`.
        * `no_mask_tokens` is a list of tokens that should not be masked.
        This is useful if we are training the MLM with another task like classification at the same time,
        and we have tokens such as `[CLS]` that shouldn't be masked.
        * `n_tokens` total number of tokens (used for generating random tokens)
        * `masking_prob` is the masking probability
        * `randomize_prob` is the probability of replacing with a random token
        * `no_change_prob` is the probability of replacing with original token
        """
        self.n_tokens = n_tokens
        self.no_change_prob = no_change_prob
        self.randomize_prob = randomize_prob
        self.masking_prob = masking_prob
        self.no_mask_tokens = no_mask_tokens + [padding_token, mask_token]
        self.padding_token = padding_token
        self.mask_token = mask_token

    def __call__(self, x: torch.Tensor):
        """
        * `x` is the batch of input token sequences.
         It's a tensor of type `long` with shape `[batch_size, seq_len]`.
        """
        shape = x.shape[:2]
        # Mask `masking_prob` of tokens
        full_mask = torch.rand(shape, device=x.device) < self.masking_prob

        # A mask for tokens to be replaced with original tokens
        unchanged = full_mask & (torch.rand(shape, device=x.device) < self.no_change_prob)
        # A mask for tokens to be replaced with a random token
        random_token_mask = full_mask & (torch.rand(shape, device=x.device) < self.randomize_prob)
        # Indexes of tokens to be replaced with random tokens
        random_token_idx = torch.nonzero(random_token_mask, as_tuple=True)
        # Random tokens for each of the locations
        random_tokens = torch.rand( (len(random_token_idx[0]),3), device=x.device, dtype=x.dtype)*2*math.pi-math.pi # -math.pi, math.pi,
        # The final set of tokens that are going to be replaced by `[MASK]`
        mask = full_mask & ~random_token_mask & ~unchanged

        # Make a clone of the input for the labels
        y = x.clone()

        # Replace with `[MASK]` tokens;
        # note that this doesn't include the tokens that will have the original token unchanged and
        # those that get replace with a random token.
        x.masked_fill_(mask[...,None].repeat(1,1,3), self.mask_token)
        # Assign random tokens
        x[random_token_idx] = random_tokens

        # Assign token `[PAD]` to all the other locations in the labels.
        # The labels equal to `[PAD]` will not be used in the loss.
        y.masked_fill_(~full_mask[...,None].repeat(1,1,3), self.padding_token)

        return x, y, full_mask