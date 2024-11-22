import torch
import torch.nn.functional as F
import os
from openfold.data import residue_constants, protein
from transformers import AutoTokenizer
from openfold.utils import data_utils as du

AA_NAME_SYM = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
            'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
            'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'UNK': "U"
        }

AA_NAME_SYM_inv = {value: key for key, value in AA_NAME_SYM.items()}

def dot(a,b):
    return torch.sum(a*b,dim=-1, keepdim=True)

def norm_vector(v):
    return v/(torch.norm(v, dim=-1,keepdim=True)+1e-8)

def get_length(atom2, atom3):
    return torch.norm(atom3-atom2, dim=-1, keepdim=True)

def get_angles(atom1, atom2, atom3):
    v1 = norm_vector(atom1-atom2)
    v2 = norm_vector(atom3-atom2)
    return torch.arccos(dot(v1,v2))

def get_dihedral(atoms1, atoms2, atoms3, atoms4):
    """
    Measure the dihedral angle between 4 atoms.
    
    Parameters
    ----------
    atoms1, atoms2, atoms3, atoms4 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the dihedral angle between.
        Alternatively an ndarray containing the coordinates can be
        provided.
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, periodic boundary conditions are
        taken into account (minimum-image convention), based on
        the box vectors given with this parameter.
        The shape *(m,3,3)* is only allowed, when the input coordinates
        comprise multiple models.
    
    Returns
    -------
    dihed : float or ndarray
        The dihedral angle(s) between the atoms. The shape is equal to
        the shape of the input `atoms` with the highest dimensionality
        minus the last axis.
    
    See Also
    --------
    index_dihedral
    dihedral_backbone
    """
    v1 = atoms2 -atoms1
    v2 = atoms3 -atoms2
    v3 = atoms4 -atoms3
    v1 = norm_vector(v1)
    v2 = norm_vector(v2)
    v3 = norm_vector(v3)
    
    n1 = torch.cross(v1, v2)
    n2 = torch.cross(v2, v3)
    
    # v1_hat = norm_vector(-v1 - dot(-v1,v2)*v2)
    # v3_hat = norm_vector(v3 - dot(v3,v2)*v2)
    
    # Calculation using atan2, to ensure the correct sign of the angle 
    delta_x = dot(n1,n2)  # n1,n2之间的cos值
    dalta_y = dot(torch.cross(n1,n2), v2) # torch.cross(n1,n2)与v2共线, 但方向不一定相同, # n1,n2之间的sin值
    return torch.arctan2(dalta_y,delta_x)


def place_dihedral(
        a,
        b,
        c,
        bond_angle,
        bond_length,
        torsion_angle,
    ):
        """
        Place the point d such that the bond angle, length, and torsion angle are satisfied
        with the series a, b, c, d.
        """
        unit_vec = lambda x: x / torch.norm(x, dim=-1, keepdim=True)
        ab = b - a
        bc = unit_vec(c - b)
        d = torch.cat(
            [
                -bond_length * torch.cos(bond_angle), #局部坐标系上x轴坐标
                bond_length * torch.cos(torsion_angle) * torch.sin(bond_angle),  #局部坐标系上y轴坐标
                bond_length * torch.sin(torsion_angle) * torch.sin(bond_angle), #局部坐标系上z轴坐标
            ],
            dim=-1
        )
        n = unit_vec(torch.cross(ab, bc))
        nbc = torch.cross(n, bc)
        m = torch.stack([bc, nbc, n], dim=-2)
        d = torch.einsum("blxn,blx->bln", m, d)
        return d + c


def compute_frenet_frames(x, mask, eps=1e-10):
    # x: [b, n_res, 3]

    t = x[:, 1:] - x[:, :-1]
    t_norm = torch.sqrt(eps + torch.sum(t ** 2, dim=-1))
    t = t / t_norm.unsqueeze(-1)

    b = torch.cross(t[:, :-1], t[:, 1:])
    b_norm = torch.sqrt(eps + torch.sum(b ** 2, dim=-1))
    b = b / b_norm.unsqueeze(-1)

    n = torch.cross(b, t[:, 1:])

    tbn = torch.stack([t[:, 1:], b, n], dim=-1)

    # TODO: recheck correctness of this implementation
    rots = []
    for i in range(mask.shape[0]):
        rots_ = torch.eye(3).unsqueeze(0).repeat(mask.shape[1], 1, 1)
        length = torch.sum(mask[i]).int()
        rots_[1:length-1] = tbn[i, :length-2]
        rots_[0] = rots_[1]
        rots_[length-1] = rots_[length-2]
        rots.append(rots_)
    rots = torch.stack(rots, dim=0).to(x.device)

    return rots

import numpy as np
class Converter:
    def __init__(self):
        super(Converter, self).__init__()
        self.rand_vector = torch.randn(3,3)
        
    def coord2angle(self, x):
        '''
        x: [N, 3] --> angles: [N, 9]
        '''
        if type(x) == np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        
        if len(x.shape)==2:
            L, _ = x.shape
            B = 1
            flatten = True
        if len(x.shape)==3:
            B, L, _ = x.shape
            flatten = False
        
        x = x.reshape(B, L, 3)
        x_extend = F.pad(x, (0,0,3,0))
        X = torch.stack([x_extend[:,0:-3], x_extend[:,1:-2], x_extend[:,2:-1], x_extend[:,3:]], dim=-2)
        
        bond_lengths = get_length(X[:,:,2],X[:,:,3])
        # bond_lengths[0] = torch.nan
        bond_angles = get_angles(X[:,:,1],X[:,:,2],X[:,:,3])
        # bond_angles[0] = torch.nan
        torsion_angles = get_dihedral(X[:,:,0], X[:,:,1], X[:,:,2], X[:,:,3])
        # torsion_angles[0:2] = torch.nan
        
        # pred_x = place_dihedral(X[:,0], X[:,1], X[:,2], bond_angles, bond_lengths, torsion_angles)
        # print(pred_x - X[:,3])

        feats = torch.cat([bond_lengths, bond_angles, torsion_angles], axis=-1)
        if flatten:
            return feats[0]
        return feats
    
    def angle2coord(self, bond_lengths, bond_angles, torsion_angles, x=None):
        '''
        [length,1]
        '''
        if len(bond_lengths.shape)==2:
            L, _ = bond_lengths.shape
            B = 1
            flatten = True
        if len(bond_lengths.shape)==3:
            B, L, _ = bond_lengths.shape
            flatten = False
        
        bond_lengths = bond_lengths.reshape(B, L, 1)
        bond_angles = bond_angles.reshape(B, L, 1)
        torsion_angles = torsion_angles.reshape(B, L, 1)
        if x is not None:
            x = x.reshape(B, -1, 3)
        
        if x is not None:
            x_init = x[:,:3]
        else:
            x_init = torch.rand(B,3,3, device=bond_lengths.device, dtype=bond_lengths.dtype)*10
        
        all_pos = []
        for i in range(0, L):
            pred_x = place_dihedral(x_init[:,0:1], x_init[:,1:2], x_init[:,2:3], bond_angles[:,i:i+1], bond_lengths[:,i:i+1], torsion_angles[:,i:i+1])
            all_pos.append(pred_x)
            x_init = torch.roll(x_init, shifts=-1, dims=-2)
            x_init[:,-1:] = pred_x
        all_pos = torch.cat(all_pos, dim=-2)
        
        if flatten:
            return all_pos[0]
        return all_pos

    def coord2frame(self,x, mask):
        '''
        x: [batch, L, 3]
        mask: [batch, L] 0,1数组
        '''
        from utils.affine_utils import T
        trans = x - torch.mean(x, dim=1, keepdim=True)
        rots = compute_frenet_frames(trans, mask)
        return T(rots, trans), mask
        
        
    
    def save2pdb(self, out_fname, coords, seqs=None):
        import biotite.structure as struc
        from biotite.structure.io.pdb import PDBFile
        
        atoms = []
        for i in range(coords.shape[0]):  
            coord = coords[i]

            b_factor = 5.0
            if seqs is None:
                atom = struc.Atom(
                    coord,
                    chain_id="A",
                    res_id=i + 1,
                    atom_id=i + 1,
                    res_name="GLY",
                    atom_name="CA",
                    element="C",
                    occupancy=1.0,
                    hetero=False,
                    b_factor=b_factor,
                )
            else:
                atom = struc.Atom(
                    coord,
                    chain_id="A",
                    res_id=i + 1,
                    atom_id=i + 1,
                    res_name=AA_NAME_SYM_inv[seqs[i]],
                    atom_name="CA",
                    element="C",
                    occupancy=1.0,
                    hetero=False,
                    b_factor=b_factor,
                )
            atoms.append(atom)
        full_structure = struc.array(atoms)
        
        # Add bonds
        full_structure.bonds = struc.BondList(full_structure.array_length())
        indices = list(range(full_structure.array_length()))
        for a, b in zip(indices[:-1], indices[1:]):
            full_structure.bonds.add_bond(a, b, bond_type=struc.BondType.SINGLE)

        sink = PDBFile()
        sink.set_structure(full_structure)
        if not os.path.exists(os.path.dirname(out_fname)):
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)
        sink.write(out_fname)
        return out_fname
        
    def Chain2AF2(self, seqs, coords, chain_id='A', b_factors=None):
        '''
        seqs: [L]
        coords: [L,3]
        '''
        chain_id = du.chain_str_to_int(chain_id)
        N = len(seqs)
        atom_positions = np.zeros((N, 37, 3))
        atom_positions[:,residue_constants.atom_order['CA']] = coords

        atom_mask = np.zeros((N, 37))
        atom_mask[:,residue_constants.atom_order['CA']] = 1

        aatype, residue_index, chain_ids = [], [], []
        for idx, res_shortname in enumerate(seqs):
            residue_index.append(idx)
            chain_ids.append(chain_id)
        aatype = self.ESM_tokenizer.encode("".join(seqs), add_special_tokens=False)
        aatype = np.array(aatype)
        residue_index = np.array(residue_index)
        chain_ids = np.array(chain_ids)

        if b_factors is None:
            b_factors = [100 for i in range(N)]

        data = protein.Protein(
            atom_positions=atom_positions, # [121,37,3]
            atom_mask=atom_mask, # [121,37]
            aatype=aatype, # [121]
            residue_index=residue_index, # [121]
            chain_index=chain_ids, # [121]
            b_factors=np.array(b_factors)) # [121,37]
        return data