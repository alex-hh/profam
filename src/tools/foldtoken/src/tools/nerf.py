"""
NERF!
Note that this was designed with compatibility with biotite, NOT biopython!
These two packages use different conventions for where NaNs are placed in dihedrals

References:
https://benjamin-computer.medium.com/protein-loops-in-tensorflow-a-i-bio-part-2-f1d802ef8300
https://www.biotite-python.org/examples/gallery/structure/peptide_assembly.html
"""
import os
from functools import cached_property
import itertools
from tkinter import N
from typing import *


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import modulo_with_wrapped_range


N_CA_LENGTH = 1.46  # Check, approxiamtely right
CA_C_LENGTH = 1.54  # Check, approximately right
C_N_LENGTH = 1.34  # Check, approximately right
C_O_LENGTH = 1.23

# Taken from initial coords from 1CRN, which is a THR
N_INIT = np.array([17.047, 14.099, 3.625])
CA_INIT = np.array([16.967, 12.784, 4.338])
C_INIT = np.array([15.685, 12.755, 5.133])
O_INIT = np.array([15.268, 13.825, 5.594])

def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor

def _normalize(tensor, dim=-1):
    return nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def rigid_transform_P2Q(P, Q, mask):
    '''
    mapping from P to Q
    P: [batch, N, 3]
    Q: [batch, N, 3]
    mask: [batch, N]
    P_hat = (R @ P.permute(0,2,1)).permute(0,2,1) + t
    '''
    # find mean column wise: 3 x 1
    centroid_P = torch.sum(P*mask[...,None], dim=-2, keepdims=True) / torch.sum(mask[...,None], dim=-2, keepdims=True)
    centroid_Q = torch.sum(Q*mask[...,None], dim=-2, keepdims=True) / torch.sum(mask[...,None], dim=-2, keepdims=True)

    # subtract mean
    Pm = P - centroid_P
    Qm = Q - centroid_Q

    H = (Pm*mask[...,None]).permute(0,2,1) @ (Qm*mask[...,None])

    # find rotation
    U, S, Vt = torch.linalg.svd(H)

    d = torch.sign(torch.linalg.det(Vt.permute(0,2,1) @ U.permute(0,2,1)))
    SS = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=1))

    R = (Vt.permute(0,2,1) @ SS) @ U.permute(0,2,1)

    t = -(R @ centroid_P.permute(0,2,1)).permute(0,2,1) + centroid_Q

    P_hat = (R @ P.permute(0,2,1)).permute(0,2,1) + t
    return P_hat


class NERFBuilder:
    """
    Builder for NERF
    """

    def __init__(
        self,
        phi_dihedrals: np.ndarray,
        psi_dihedrals: np.ndarray,
        omega_dihedrals: np.ndarray,
        bond_len_n_ca: Union[float, np.ndarray] = N_CA_LENGTH,
        bond_len_ca_c: Union[float, np.ndarray] = CA_C_LENGTH,
        bond_len_c_n: Union[float, np.ndarray] = C_N_LENGTH,  # 0C:1N distance
        bond_angle_n_ca: Union[float, np.ndarray] = 121 / 180 * np.pi,
        bond_angle_ca_c: Union[float, np.ndarray] = 109 / 180 * np.pi,  # aka tau
        bond_angle_c_n: Union[float, np.ndarray] = 115 / 180 * np.pi,
        init_coords: np.ndarray = [N_INIT, CA_INIT, C_INIT],
    ) -> None:
        self.phi = phi_dihedrals.squeeze()
        self.psi = psi_dihedrals.squeeze()
        self.omega = omega_dihedrals.squeeze()

        # We start with coordinates for N --> CA --> C so the next atom we add
        # is the next N. Therefore, the first angle we need is the C --> N bond
        self.bond_lengths = {
            ("C", "N"): bond_len_c_n,
            ("N", "CA"): bond_len_n_ca,
            ("CA", "C"): bond_len_ca_c,
        }
        self.bond_angles = {
            ("C", "N"): bond_angle_c_n,
            ("N", "CA"): bond_angle_n_ca,
            ("CA", "C"): bond_angle_ca_c,
        }
        self.init_coords = [c.squeeze() for c in init_coords]
        assert len(self.init_coords) == 3, f"Requires 3 initial coords for N-Ca-C but got {len(self.init_coords)}"
        assert all([c.size == 3 for c in self.init_coords]), "Initial coords should be 3-dimensional"

        self.bonds = itertools.cycle(self.bond_angles.keys())

    @cached_property
    def cartesian_coords(self) -> np.ndarray:
        """Build out the molecule"""
        retval = self.init_coords.copy()

        # The first value of phi at the N terminus is not defined
        # The last value of psi and omega at the C terminus are not defined
        for i, (phi, psi, omega) in enumerate(
            zip(self.phi[1:], self.psi[:-1], self.omega[:-1])
        ):
            # Procedure for placing N-CA-C
            # Place the next N atom, which requires the C-N bond length/angle, and the psi dihedral
            # Place the alpha carbon, which requires the N-CA bond length/angle, and the omega dihedral
            # Place the carbon, which requires the the CA-C bond length/angle, and the phi dihedral
            for bond, dih in zip(self.bond_lengths.keys(), [psi, omega, phi]):
                coords = self.place_dihedral(
                    retval[-3],
                    retval[-2],
                    retval[-1],
                    bond_angle=self._get_bond_angle(bond, i),
                    bond_length=self._get_bond_length(bond, i),
                    torsion_angle=dih,
                )
                retval.append(coords)

        return np.array(retval)

    @cached_property
    def centered_cartesian_coords(self) -> np.ndarray:
        """Returns the centered coords"""
        means = self.cartesian_coords.mean(axis=0)
        return self.cartesian_coords - means

    def _get_bond_length(self, bond: Tuple[str, str], idx: int):
        """Get the ith bond distance"""
        v = self.bond_lengths[bond]
        if isinstance(v, float):
            return v
        return v[idx]

    def _get_bond_angle(self, bond: Tuple[str, str], idx: int):
        """Get the ith bond angle"""
        v = self.bond_angles[bond]
        if isinstance(v, float):
            return v
        return v[idx]


    def place_dihedral(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        bond_angle: float,
        bond_length: float,
        torsion_angle: float,
    ) -> np.ndarray:
        """
        Place the point d such that the bond angle, length, and torsion angle are satisfied
        with the series a, b, c, d.
        """
        assert a.ndim == b.ndim == c.ndim == 1
        unit_vec = lambda x: x / np.linalg.norm(x)
        ab = b - a
        bc = unit_vec(c - b)
        d = np.array(
            [
                -bond_length * np.cos(bond_angle),
                bond_length * np.cos(torsion_angle) * np.sin(bond_angle),
                bond_length * np.sin(torsion_angle) * np.sin(bond_angle),
            ]
        )
        n = unit_vec(np.cross(ab, bc))
        nbc = np.cross(n, bc)
        m = np.stack([bc, nbc, n]).T
        d = m.dot(d)
        return d + c


class TorchNERFBuilder_Oxygen(nn.Module):
    """
    Builder for NERF
    """

    def __init__(
        self,
        phi_dihedrals,
        psi_dihedrals,
        omega_dihedrals,
        O_dihedrals,
        bond_len_n_ca: Union[float, np.ndarray] = N_CA_LENGTH,
        bond_len_ca_c: Union[float, np.ndarray] = CA_C_LENGTH,
        bond_len_c_n: Union[float, np.ndarray] = C_N_LENGTH,  # 0C:1N distance
        bond_len_c_o = C_O_LENGTH,
        bond_angle_n_ca: Union[float, np.ndarray] = 121 / 180 * np.pi,
        bond_angle_ca_c: Union[float, np.ndarray] = 109 / 180 * np.pi,  # aka tau
        bond_angle_c_n: Union[float, np.ndarray] = 115 / 180 * np.pi,
        bond_angle_c_o: Union[float, np.ndarray] = 120.5 / 180 * np.pi,
        init_coords = [N_INIT, CA_INIT, C_INIT, O_INIT],
        virtual_num = 3
    ) -> None:
        super(TorchNERFBuilder_Oxygen, self).__init__()
        self.phi = phi_dihedrals.squeeze()
        self.psi = psi_dihedrals.squeeze()
        self.omega = omega_dihedrals.squeeze()
        self.O_dihedrals = O_dihedrals.squeeze()

        # We start with coordinates for N --> CA --> C so the next atom we add
        # is the next N. Therefore, the first angle we need is the C --> N bond
        self.bond_lengths = {
            ("C", "N"): bond_len_c_n,
            ("N", "CA"): bond_len_n_ca,
            ("CA", "C"): bond_len_ca_c,
            ("C", "O"): bond_len_c_o
        }
        self.bond_angles = {
            ("C", "N"): bond_angle_c_n,
            ("N", "CA"): bond_angle_n_ca,
            ("CA", "C"): bond_angle_ca_c,
            ("C", "O"): bond_angle_c_o
        }
        batch = psi_dihedrals.shape[0]
        self.batch = batch
        self.device = phi_dihedrals.device
        a = torch.tensor(init_coords[0], device=self.device).repeat(batch,1).float()
        b = torch.tensor(init_coords[1], device=self.device).repeat(batch,1).float()
        c = torch.tensor(init_coords[2], device=self.device).repeat(batch,1).float()
        d = torch.tensor(init_coords[3], device=self.device).repeat(batch,1).float()
        self.init_coords = [a, b, c, d]
        self.virtual_num = virtual_num
        self.virtual_atoms = nn.Parameter(torch.rand(self.virtual_num,3))

        # self.bonds = itertools.cycle(self.bond_angles.keys())

    @cached_property
    def cartesian_coords(self) -> np.ndarray:
        """Build out the molecule"""
        retval = self.init_coords.copy()

        # The first value of phi at the N terminus is not defined
        # The last value of psi and omega at the C terminus are not defined
        for i in range(self.phi.shape[1]-1):
            phi, psi, omega, O_dihedral = self.phi[:,1+i].view(-1,1), self.psi[:,i].view(-1,1), self.omega[:,i].view(-1,1), self.O_dihedrals[:,i].view(-1,1)
            # Procedure for placing N-CA-C-O
            # Place the next N atom, which requires the C-N bond length/angle, and the psi dihedral
            # Place the alpha carbon, which requires the N-CA bond length/angle, and the omega dihedral
            # Place the carbon, which requires the the CA-C bond length/angle, and the phi dihedral
            # Place the Oxygen, which requires the the CA-C bond length/angle, and the O_dihedral
            for bond, dih in zip(self.bond_lengths.keys(), [psi, omega, phi, O_dihedral]):
                if bond in [("C", "N")]:
                    coords = self.place_dihedral(
                        retval[-4],
                        retval[-3],
                        retval[-2],
                        bond_angle=self._get_bond_angle(bond, i),
                        bond_length=self._get_bond_length(bond, i),
                        torsion_angle=dih,
                    )
                elif bond in [("N", "CA")]:
                    coords = self.place_dihedral(
                        retval[-4],
                        retval[-3],
                        retval[-1],
                        bond_angle=self._get_bond_angle(bond, i),
                        bond_length=self._get_bond_length(bond, i),
                        torsion_angle=dih,
                    )
                if bond in [("CA", "C")]:
                    coords = self.place_dihedral(
                        retval[-4],
                        retval[-2],
                        retval[-1],
                        bond_angle=self._get_bond_angle(bond, i),
                        bond_length=self._get_bond_length(bond, i),
                        torsion_angle=dih,
                    )
                if bond in [("C", "O")]:
                    coords = self.place_dihedral(
                        retval[-3],
                        retval[-2],
                        retval[-1],
                        bond_angle=self._get_bond_angle(bond, i),
                        bond_length=self._get_bond_length(bond, i),
                        torsion_angle=dih,
                    )
                retval.append(coords)

        return torch.stack(retval, dim=1)

    @cached_property
    def centered_cartesian_coords(self) -> np.ndarray:
        """Returns the centered coords"""
        means = self.cartesian_coords.mean(axis=0)
        return self.cartesian_coords - means

    def _get_bond_length(self, bond: Tuple[str, str], idx: int):
        """Get the ith bond distance"""
        v = self.bond_lengths[bond]
        if isinstance(v, float):
            return torch.tensor(v, device=self.device).repeat(self.batch,1)
        if len(v.shape)==2:
            return v[:,idx].reshape(-1,1)
        return v[idx]

    def _get_bond_angle(self, bond: Tuple[str, str], idx: int):
        """Get the ith bond angle"""
        v = self.bond_angles[bond]
        if isinstance(v, float):
            return torch.tensor(v, device=self.device).repeat(self.batch,1)
        if len(v.shape)==2:
            return v[:,idx].reshape(-1,1)
        return v[idx]


    def place_dihedral(
        self,
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
        m = torch.stack([bc, nbc, n], dim=1)
        d = torch.einsum("bxn,bx->bn", m, d)
        return d + c

    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx  

    @classmethod
    def _get_dihedral(self, source_struct):
        import biotite.structure as struc
        from biotite.structure import filter_amino_acids, chain_iter, angle, distance, dihedral
        phi, psi, omega = struc.dihedral_backbone(source_struct)
        bb_filter = (   ((source_struct.atom_name == "N") |
                    (source_struct.atom_name == "CA") |
                    (source_struct.atom_name == "C") |
                    (source_struct.atom_name == "O")) &
                    filter_amino_acids(source_struct) )
        backbone = source_struct[..., bb_filter]
        chain_bb = backbone[0]
        bb_coord = chain_bb.coord # [N,CA,C,O]
        bb_coord = bb_coord.reshape(1,-1,4,3)
        N_CA_C_O = dihedral(bb_coord[:,1:,0,:], bb_coord[:,1:,1,:], bb_coord[:,1:,2,:], bb_coord[:,1:,3,:])
        phi = torch.tensor(phi).repeat(4,1)
        psi = torch.tensor(psi).repeat(4,1)
        omega = torch.tensor(omega).repeat(4,1)
        N_CA_C_O = torch.tensor(N_CA_C_O).repeat(4,1)
        return phi, psi, omega, N_CA_C_O, bb_coord
    
    @classmethod
    def _get_bond_angles(self, source_struct):
        import biotite.structure as struc
        from biotite.structure import filter_amino_acids, chain_iter, angle, distance
        bb_filter = (   ((source_struct.atom_name == "N") |
                    (source_struct.atom_name == "CA") |
                    (source_struct.atom_name == "C") |
                    (source_struct.atom_name == "O")) &
                    filter_amino_acids(source_struct) )
        backbone = source_struct[..., bb_filter]
        chain_bb = backbone[0]
        bb_coord = chain_bb.coord # [N,CA,C,O]
        bb_coord = bb_coord.reshape(1,-1,4,3)

        CA_C_N = angle(bb_coord[:,:-1,1,:], bb_coord[:,:-1,2,:], bb_coord[:,1:,0,:])
        C_N_CA = angle(bb_coord[:,:-1,2,:], bb_coord[:,1:,0,:], bb_coord[:,1:,1,:])
        N_CA_C = angle(bb_coord[:,1:,0,:], bb_coord[:,1:,1,:], bb_coord[:,1:,2,:])
        CA_C_O = angle(bb_coord[:,1:,1,:], bb_coord[:,1:,2,:], bb_coord[:,1:,3,:])
        

        CA_C_N = torch.tensor(CA_C_N).repeat(4,1)
        C_N_CA = torch.tensor(C_N_CA).repeat(4,1)
        N_CA_C = torch.tensor(N_CA_C).repeat(4,1)
        CA_C_O = torch.tensor(CA_C_O).repeat(4,1)
        return CA_C_N, C_N_CA, N_CA_C, CA_C_O
    
    @classmethod
    def _get_bond_lengths(self, source_struct):
        import biotite.structure as struc
        from biotite.structure import filter_amino_acids, chain_iter, angle, distance
        bb_filter = (   ((source_struct.atom_name == "N") |
                    (source_struct.atom_name == "CA") |
                    (source_struct.atom_name == "C") |
                    (source_struct.atom_name == "O")) &
                    filter_amino_acids(source_struct) )
        backbone = source_struct[..., bb_filter]
        chain_bb = backbone[0]
        bb_coord = chain_bb.coord # [N,CA,C,O]
        bb_coord = bb_coord.reshape(1,-1,4,3)

        C_N = distance(bb_coord[:,:-1,2,:], bb_coord[:,1:,0,:])
        N_CA = distance(bb_coord[:,1:,0,:], bb_coord[:,1:,1,:])
        CA_C = distance(bb_coord[:,1:,1,:], bb_coord[:,1:,2,:])
        C_O = distance( bb_coord[:,1:,2,:], bb_coord[:,1:,3,:])
        
        C_N = torch.tensor(C_N).repeat(4,1)
        N_CA = torch.tensor(N_CA).repeat(4,1)
        CA_C = torch.tensor(CA_C).repeat(4,1)
        C_O = torch.tensor(C_O).repeat(4,1)
        return C_N, N_CA, CA_C, C_O


    @classmethod
    def sv2pdb(
        self,
        out_fname: str,
        coords,
        unknown_mask = None
    ) -> str:
        """
        Create a new chain using NERF to convert to cartesian coordinates. Returns
        the path to the newly create file if successful, empty string if fails.
        """
        import biotite.structure as struc
        from biotite.structure.io.pdb import PDBFile
        # Create a new PDB file using biotite
        # https://www.biotite-python.org/tutorial/target/index.html#creating-structures
        atoms = []
        for i, (n_coord, ca_coord, c_coord, o_coord) in enumerate(
            (coords[j : j + 4] for j in range(0, len(coords), 4))
        ):  
            if unknown_mask is not None:
                b_factor = unknown_mask[i]*100 + ~unknown_mask[i]*5
            else:
                b_factor = 5.0
                
            atom1 = struc.Atom(
                n_coord,
                chain_id="A",
                res_id=i + 1,
                atom_id=i * 4 + 1,
                res_name="GLY",
                atom_name="N",
                element="N",
                occupancy=1.0,
                hetero=False,
                b_factor=b_factor,
            )
            atom2 = struc.Atom(
                ca_coord,
                chain_id="A",
                res_id=i + 1,
                atom_id=i * 4 + 2,
                res_name="GLY",
                atom_name="CA",
                element="C",
                occupancy=1.0,
                hetero=False,
                b_factor=b_factor,
            )
            atom3 = struc.Atom(
                c_coord,
                chain_id="A",
                res_id=i + 1,
                atom_id=i * 4 + 3,
                res_name="GLY",
                atom_name="C",
                element="C",
                occupancy=1.0,
                hetero=False,
                b_factor=b_factor,
            )
            atom4 = struc.Atom(
                o_coord,
                chain_id="A",
                res_id=i + 1,
                atom_id=i * 4 + 4,
                res_name="GLY",
                atom_name="O",
                element="O",
                occupancy=1.0,
                hetero=False,
                b_factor=b_factor,
            )
            atoms.extend([atom1, atom2, atom3, atom4])
        full_structure = struc.array(atoms)

        # Add bonds
        full_structure.bonds = struc.BondList(full_structure.array_length())
        indices = list(range(full_structure.array_length()))
        for a, b in zip(indices[:-1], indices[1:]):
            full_structure.bonds.add_bond(a, b, bond_type=struc.BondType.SINGLE)

        # Annotate secondary structure using CA coordinates
        # https://www.biotite-python.org/apidoc/biotite.structure.annotate_sse.html
        # https://academic.oup.com/bioinformatics/article/13/3/291/423201
        # a = alpha helix, b = beta sheet, c = coil
        # ss = struc.annotate_sse(full_structure, "A")
        # full_structure.set_annotation("secondary_structure_psea", ss)

        sink = PDBFile()
        sink.set_structure(full_structure)
        sink.write(out_fname)
        return out_fname



class TorchNERFBuilder(nn.Module):
    """
    Builder for NERF
    """

    def __init__(
        self,
        virtual_num = 3,
        num_rbf = 16
    ) -> None:
        super(TorchNERFBuilder, self).__init__()
        self.virtual_num = virtual_num
        self.num_rbf = num_rbf
        
        self.AA_NAME_SYM = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
            'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
            'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'UNK': "U"
        }
        
        self.AA_SYM_NAME2 = {v:k for k,v in self.AA_NAME_SYM.items()}
        # self.virtual_atoms = nn.Parameter(torch.rand(self.virtual_num,3))
        

    def set_values(
        self, 
        phi_dihedrals,
        psi_dihedrals,
        omega_dihedrals,
        bond_len_n_ca: Union[float, np.ndarray] = N_CA_LENGTH,
        bond_len_ca_c: Union[float, np.ndarray] = CA_C_LENGTH,
        bond_len_c_n: Union[float, np.ndarray] = C_N_LENGTH,  # 0C:1N distance
        bond_angle_n_ca: Union[float, np.ndarray] = 121 / 180 * np.pi,
        bond_angle_ca_c: Union[float, np.ndarray] = 109 / 180 * np.pi,  # aka tau
        bond_angle_c_n: Union[float, np.ndarray] = 115 / 180 * np.pi,
        res_types = None,
        init_coords = [N_INIT, CA_INIT, C_INIT],
        max_length = 128
        ):
        self.phi = phi_dihedrals
        self.psi = psi_dihedrals
        self.omega = omega_dihedrals
        self.res_type = res_types
        # We start with coordinates for N --> CA --> C so the next atom we add
        # is the next N. Therefore, the first angle we need is the C --> N bond
        self.bond_lengths = {
            ("C", "N"): bond_len_c_n,
            ("N", "CA"): bond_len_n_ca,
            ("CA", "C"): bond_len_ca_c
        }
        self.bond_angles = {
            ("C", "N"): bond_angle_c_n,
            ("N", "CA"): bond_angle_n_ca,
            ("CA", "C"): bond_angle_ca_c
        }
        batch = psi_dihedrals.shape[0]
        self.batch = batch
        self.device = phi_dihedrals.device
        
        if len(init_coords[0].shape)==1:
            a = torch.tensor(init_coords[0], device=self.device).repeat(batch,1).float()
            b = torch.tensor(init_coords[1], device=self.device).repeat(batch,1).float()
            c = torch.tensor(init_coords[2], device=self.device).repeat(batch,1).float()
            self.init_coords = [a, b, c]
        else:
            self.init_coords = [init_coords[:,0,:], init_coords[:,1,:], init_coords[:,2,:]]
        self.max_length = max_length

    def cartesian_coords(self):
        """Build out the molecule"""
        retval = self.init_coords.copy()

        # The first value of phi at the N terminus is not defined
        # The last value of psi and omega at the C terminus are not defined
        for i in range(min(self.phi.shape[1]-1, self.max_length-1)):
            psi = self.psi[:,i].view(-1,1) # for N_{i+1}
            omega = self.omega[:,i].view(-1,1) # for CA_{i+1}
            phi = self.phi[:,i+1].view(-1,1) # for C_{i+1}
            
            # Procedure for placing N-CA-C
            # Place the next N atom, which requires the C-N bond length/angle, and the psi dihedral
            # Place the alpha carbon, which requires the N-CA bond length/angle, and the omega dihedral
            # Place the carbon, which requires the the CA-C bond length/angle, and the phi dihedral
            # Place the Oxygen, which requires the the CA-C bond length/angle, and the O_dihedral
            for bond, dih in zip(self.bond_lengths.keys(), [psi, omega, phi]):
                coords = self.place_dihedral(
                    retval[-3],
                    retval[-2],
                    retval[-1],
                    bond_angle=self._get_bond_angle(bond, i),
                    bond_length=self._get_bond_length(bond, i),
                    torsion_angle=dih,
                )
                retval.append(coords)
        pred = torch.stack(retval, dim=1) 
        '''
        在初始化角度的时候, 默认为nan, 当序列长度不够batch中的最大长度时, padding部分的坐标为nan:/huyuqi/xmyu/DiffSDS/tools/convertor.py---get_dihedrals()函数
        这可能会导致后续计算产生nan。毕竟nan乘以0依旧为nan, 仅仅用乘法做mask是不够的.
        '''
        
        nan_mask = torch.isnan(pred)
        pred[nan_mask] = 0
        return pred

    @cached_property
    def centered_cartesian_coords(self) -> np.ndarray:
        """Returns the centered coords"""
        means = self.cartesian_coords.mean(axis=0)
        return self.cartesian_coords - means

    def _get_bond_length(self, bond: Tuple[str, str], idx: int):
        """Get the ith bond distance"""
        v = self.bond_lengths[bond]
        if isinstance(v, float):
            return torch.tensor(v, device=self.device).repeat(self.batch,1)
        if len(v.shape)==2:
            return v[:,idx].reshape(-1,1)
        return v[idx]

    def _get_bond_angle(self, bond: Tuple[str, str], idx: int):
        """Get the ith bond angle"""
        v = self.bond_angles[bond]
        if isinstance(v, float):
            return torch.tensor(v, device=self.device).repeat(self.batch,1)
        if len(v.shape)==2 or len(v.shape)==3:
            if bond==("C", "N"):
                return v[:,idx].reshape(-1,1)
            elif bond==("N", "CA"):
                return v[:,idx].reshape(-1,1)
            elif bond==("CA", "C"):
                return v[:,idx].reshape(-1,1)


    def place_dihedral(
        self,
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
                -bond_length * torch.cos(bond_angle),
                bond_length * torch.cos(torsion_angle) * torch.sin(bond_angle),
                bond_length * torch.sin(torsion_angle) * torch.sin(bond_angle),
            ],
            dim=-1
        )
        n = unit_vec(torch.cross(ab, bc))
        nbc = torch.cross(n, bc)
        m = torch.stack([bc, nbc, n], dim=1) # [batch, number of vectors, vector values]
        d = torch.einsum("bnd,bn->bd", m, d)
        return d + c

    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx  

    @classmethod
    def _get_dihedral(self, source_struct):
        import biotite.structure as struc
        from biotite.structure import filter_amino_acids, chain_iter, angle, distance, dihedral
        phi, psi, omega = struc.dihedral_backbone(source_struct)
        bb_filter = (   ((source_struct.atom_name == "N") |
                    (source_struct.atom_name == "CA") |
                    (source_struct.atom_name == "C")) &
                    filter_amino_acids(source_struct) )
        backbone = source_struct[..., bb_filter]
        chain_bb = backbone[0]
        bb_coord = chain_bb.coord # [N,CA,C]
        bb_coord = bb_coord.reshape(1,-1,3,3)

        phi = torch.tensor(phi).repeat(4,1)
        psi = torch.tensor(psi).repeat(4,1)
        omega = torch.tensor(omega).repeat(4,1)
        return phi, psi, omega, bb_coord
    
    @classmethod
    def _get_bond_angles(self, source_struct):
        import biotite.structure as struc
        from biotite.structure import filter_amino_acids, chain_iter, angle, distance
        bb_filter = (   ((source_struct.atom_name == "N") |
                    (source_struct.atom_name == "CA") |
                    (source_struct.atom_name == "C")) &
                    filter_amino_acids(source_struct) )
        backbone = source_struct[..., bb_filter]
        chain_bb = backbone[0]
        bb_coord = chain_bb.coord # [N,CA,C]
        bb_coord = bb_coord.reshape(1,-1,3,3)

        CA_C_N = angle(bb_coord[:,:-1,1,:], bb_coord[:,:-1,2,:], bb_coord[:,1:,0,:]) # CA, C, N,  CA-C, N-C
        C_N_CA = angle(bb_coord[:,:-1,2,:], bb_coord[:,1:,0,:], bb_coord[:,1:,1,:])
        N_CA_C = angle(bb_coord[:,1:,0,:], bb_coord[:,1:,1,:], bb_coord[:,1:,2,:])
        

        CA_C_N = torch.tensor(CA_C_N).repeat(4,1)
        C_N_CA = torch.tensor(C_N_CA).repeat(4,1)
        N_CA_C = torch.tensor(N_CA_C).repeat(4,1)
        return CA_C_N, C_N_CA, N_CA_C
    
    @classmethod
    def _get_bond_lengths(self, source_struct):
        import biotite.structure as struc
        from biotite.structure import filter_amino_acids, chain_iter, angle, distance
        bb_filter = (   ((source_struct.atom_name == "N") |
                    (source_struct.atom_name == "CA") |
                    (source_struct.atom_name == "C") ) &
                    filter_amino_acids(source_struct) )
        backbone = source_struct[..., bb_filter]
        chain_bb = backbone[0]
        bb_coord = chain_bb.coord # [N,CA,C]
        bb_coord = bb_coord.reshape(1,-1,3,3)

        C_N = distance(bb_coord[:,:-1,2,:], bb_coord[:,1:,0,:])
        N_CA = distance(bb_coord[:,1:,0,:], bb_coord[:,1:,1,:])
        CA_C = distance(bb_coord[:,1:,1,:], bb_coord[:,1:,2,:])
        
        C_N = torch.tensor(C_N).repeat(4,1)
        N_CA = torch.tensor(N_CA).repeat(4,1)
        CA_C = torch.tensor(CA_C).repeat(4,1)
        return C_N, N_CA, CA_C



    def sv2pdb(
        self,
        out_fname: str,
        coords,
        unknown_mask = None,
        attn_mask = None
    ) -> str:
        """
        Create a new chain using NERF to convert to cartesian coordinates. Returns
        the path to the newly create file if successful, empty string if fails.
        """
        import biotite.structure as struc
        from biotite.structure.io.pdb import PDBFile
        # Create a new PDB file using biotite
        # https://www.biotite-python.org/tutorial/target/index.html#creating-structures
        atoms = []
        for i, (n_coord, ca_coord, c_coord) in enumerate(
            (coords[j : j + 3] for j in range(0, len(coords), 3))
        ):
            if unknown_mask is not None:
                if unknown_mask[i]:
                    b_factor = 100
                else:
                    if attn_mask[i]==0:
                        b_factor = 0
                    else:
                        b_factor = 5
            else:
                b_factor = 5.0
            
            if self.res_type is None:
                restype = 'GLY'
            else:
                restype = self.AA_SYM_NAME2[self.res_type[i]]
                
            atom1 = struc.Atom(
                n_coord,
                chain_id="A",
                res_id=i + 1,
                atom_id=i * 3 + 1,
                res_name=restype,
                atom_name="N",
                element="N",
                occupancy=1.0,
                hetero=False,
                b_factor=b_factor,
            )
            atom2 = struc.Atom(
                ca_coord,
                chain_id="A",
                res_id=i + 1,
                atom_id=i * 3 + 2,
                res_name=restype,
                atom_name="CA",
                element="C",
                occupancy=1.0,
                hetero=False,
                b_factor=b_factor,
            )
            atom3 = struc.Atom(
                c_coord,
                chain_id="A",
                res_id=i + 1,
                atom_id=i * 3 + 3,
                res_name=restype,
                atom_name="C",
                element="C",
                occupancy=1.0,
                hetero=False,
                b_factor=b_factor,
            )
            atoms.extend([atom1, atom2, atom3])
        full_structure = struc.array(atoms)

        # Add bonds
        full_structure.bonds = struc.BondList(full_structure.array_length())
        indices = list(range(full_structure.array_length()))
        for a, b in zip(indices[:-1], indices[1:]):
            full_structure.bonds.add_bond(a, b, bond_type=struc.BondType.SINGLE)

        # Annotate secondary structure using CA coordinates
        # https://www.biotite-python.org/apidoc/biotite.structure.annotate_sse.html
        # https://academic.oup.com/bioinformatics/article/13/3/291/423201
        # a = alpha helix, b = beta sheet, c = coil
        # ss = struc.annotate_sse(full_structure, "A")
        # full_structure.set_annotation("secondary_structure_psea", ss)

        sink = PDBFile()
        sink.set_structure(full_structure)
        sink.write(out_fname)
        return out_fname

    def transform(self, pred_coords, pred_start, pred_end, fix_start, fix_end):
        '''
        pred_coords: [batch, N, 3]
        pred_start: [batch, 3]
        pred_end: [batch, 3]
        fix_start: [batch, 3]
        fix_end: [batch, 3]
        '''
        batch = pred_coords.shape[0]
        # (batch, n, 3) 方向向量
        unit_vec = lambda x: x / torch.norm(x, dim=-1, keepdim=True)
        direct1 = unit_vec(pred_end-pred_start)
        direct2 = unit_vec(fix_end-fix_start)
        H = torch.einsum("bnc,bnd->bcd", direct1, direct2)
        # find rotation
        U, S, Vt = torch.linalg.svd(H)
        d = torch.sign(torch.linalg.det(Vt.permute(0,2,1) @ U.permute(0,2,1)))
        SS = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=1))
        R = (Vt.permute(0,2,1) @ SS) @ U.permute(0,2,1)

        pred_center = (pred_start + pred_end)/2
        translation = (fix_start + fix_end)/2
        pred_coords = pred_coords - pred_center.mean(dim=1,keepdim=True)

        pred_coords = (R @ pred_coords.permute(0,2,1)).permute(0,2,1) + translation.mean(dim=1,keepdim=True)
        return pred_coords

    def _quaternions(self, R):
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
                Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[...,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        return _normalize(Q, dim=-1)

    def Gradient(self, fix_start, fix_end, end_idx=-1):
        self.psi.requires_grad = True
        self.phi.requires_grad = True
        self.omega.requires_grad = True
        self.bond_angles[("C", "N")].requires_grad = True
        self.bond_angles[("N", "CA")].requires_grad = True
        self.bond_angles[("CA", "C")].requires_grad = True
        pred_coords = self.cartesian_coords()
        batch = pred_coords.shape[0]
        coord_reshape = pred_coords.reshape(batch,-1,3,3)
        pred_start = coord_reshape[:,0]
        idx = torch.arange(end_idx.shape[0], device = end_idx.device)
        pred_end = coord_reshape[idx, end_idx]
        
        ## length loss
        length_pred = torch.norm(pred_start.mean(dim=1) - pred_end.mean(dim=1), dim=-1)
        length_true = torch.norm(fix_start.mean(dim=1) - fix_end.mean(dim=1), dim=-1)
        length_loss = torch.mean((length_pred - length_true)**2)

        ## oreintation loss
        def get_coord_system(atoms):
            N = atoms[:,0]
            CA = atoms[:,1]
            C = atoms[:,2]

            unit_vec = lambda x: x / torch.norm(x, dim=-1, keepdim=True)
            u = CA-N
            v = C-CA
            b = unit_vec(u-v)
            n = unit_vec(torch.cross(u,v))
            O = torch.stack((b, n, torch.cross(b, n)), 2) # [batch, coords, axis]
            return O

        O_start_pred = get_coord_system(pred_start)
        O_end_pred = get_coord_system(pred_end)
        O_start_true = get_coord_system(fix_start)
        O_end_true = get_coord_system(fix_end)

        R_pred = torch.matmul(O_start_pred.permute(0,2,1), O_end_pred)
        R_true = torch.matmul(O_start_true.permute(0,2,1), O_end_true)
        q_pred = self._quaternions(R_pred)
        q_true = self._quaternions(R_true)

        ## rotation loss
        rotation_loss = torch.norm(q_pred-q_true, dim=1).mean()

        loss = length_loss + rotation_loss*1000
        loss.backward()
        print("length loss: {}\trot loss: {}".format(length_loss, rotation_loss))

        grad = {"psi": self.psi.grad.clone(),
                "phi": self.phi.grad.clone(),
                "omega": self.omega.grad.clone(),
                "bond_angle_ca_c": self.bond_angles[("CA", "C")].grad.clone(),
                "bond_angle_c_n": self.bond_angles[("C", "N")].grad.clone(),
                "bond_angle_n_ca": self.bond_angles[("N", "CA")].grad.clone()
        }

        self.psi.grad.data.zero_()
        self.phi.grad.data.zero_()
        self.omega.grad.data.zero_()
        self.bond_angles[("CA", "C")].grad.data.zero_()
        self.bond_angles[("C", "N")].grad.data.zero_()
        self.bond_angles[("N", "CA")].grad.data.zero_()

        self.psi.requires_grad = False
        self.phi.requires_grad = False
        self.omega.requires_grad = False
        self.bond_angles[("C", "N")].requires_grad = False
        self.bond_angles[("N", "CA")].requires_grad = False
        self.bond_angles[("CA", "C")].requires_grad = False

        return grad



def test_nerf():
    """On the fly testing"""
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure import filter_amino_acids, chain_iter, angle, distance, dihedral

    source = PDBFile.read(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "/gaozhangyang/experiments/ProreinBinder/data/cath/dompdb/1a0aA00")
    )
    source_struct = source.get_structure()
    phi, psi, omega, bb_coord = TorchNERFBuilder._get_dihedral(source_struct)
    CA_C_N, C_N_CA, N_CA_C = TorchNERFBuilder._get_bond_angles(source_struct)
    C_N, N_CA, CA_C = TorchNERFBuilder._get_bond_lengths(source_struct)

    bb_coord = torch.from_numpy(bb_coord).repeat(4,1,1,1)
    start_idx = 10
    end_idx = 39
    builder = TorchNERFBuilder(0)
    builder.set_values( phi[:,start_idx:], 
                        psi[:,start_idx:], 
                        omega[:,start_idx:], 
                        bond_angle_n_ca = C_N_CA[:,start_idx:], 
                        bond_angle_ca_c = N_CA_C[:,start_idx:],
                        bond_angle_c_n = CA_C_N[:,start_idx:],
                        max_length = 30,
                        init_coords=bb_coord[0,start_idx])
    pred_coords = builder.cartesian_coords()
    pred_start = pred_coords[:, :3 ,:]
    pred_end = pred_coords[:, -3: ,:]
    fix_start = bb_coord[:,start_idx]
    fix_end = bb_coord[:,end_idx]
    pred_coords = builder.transform(pred_coords, pred_start, pred_end, fix_start, fix_end)

    TorchNERFBuilder.sv2pdb("./test.pdb", pred_coords[0])
    TorchNERFBuilder.sv2pdb("./raw.pdb", bb_coord[0].reshape(-1,3))
    print(builder.cartesian_coords())


def test_gradient():
    """On the fly testing"""
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure import filter_amino_acids, chain_iter, angle, distance, dihedral

    source = PDBFile.read(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "/gaozhangyang/experiments/ProreinBinder/data/cath/dompdb/1a0aA00")
    )
    source_struct = source.get_structure()
    phi, psi, omega, bb_coord = TorchNERFBuilder._get_dihedral(source_struct)
    CA_C_N, C_N_CA, N_CA_C = TorchNERFBuilder._get_bond_angles(source_struct)
    C_N, N_CA, CA_C = TorchNERFBuilder._get_bond_lengths(source_struct)

    bb_coord = torch.from_numpy(bb_coord).repeat(4,1,1,1)
    start_idx = 10
    end_idx = 39
    builder = TorchNERFBuilder(0)
    fix_start = bb_coord[:,start_idx]
    fix_end = bb_coord[:,end_idx]

    phi = phi + torch.rand_like(phi)
    psi = psi + torch.rand_like(psi)
    omega = omega + torch.rand_like(omega)
    CA_C_N = CA_C_N + torch.rand_like(CA_C_N)
    C_N_CA = C_N_CA + torch.rand_like(C_N_CA)
    N_CA_C = N_CA_C + torch.rand_like(N_CA_C)

    phi = phi[:,start_idx:]
    psi = psi[:,start_idx:]
    omega = omega[:,start_idx:]
    C_N_CA = C_N_CA[:,start_idx:]
    N_CA_C = N_CA_C[:,start_idx:]
    CA_C_N = CA_C_N[:,start_idx:]

    step_lamda = 0.0001
    steps = 300
    for it in range(steps):
        step_lamda = 0.0001*(steps-it)/steps
        builder.set_values( phi, 
                            psi, 
                            omega, 
                            bond_angle_n_ca = C_N_CA, 
                            bond_angle_ca_c = N_CA_C,
                            bond_angle_c_n = CA_C_N,
                            max_length = 30,
                            init_coords=bb_coord[0,start_idx])
        grad = builder.Gradient(fix_start, fix_end)
        phi = phi-grad['phi']*step_lamda
        psi = psi-grad['psi']*step_lamda
        omega = omega-grad['omega']*step_lamda
        C_N_CA = (C_N_CA-grad['bond_angle_n_ca']*step_lamda).detach()
        N_CA_C = (N_CA_C-grad['bond_angle_ca_c']*step_lamda).detach()
        CA_C_N = (CA_C_N-grad['bond_angle_c_n']*step_lamda).detach()

        phi = modulo_with_wrapped_range(phi, range_min=-torch.pi, range_max=torch.pi)
        psi = modulo_with_wrapped_range(psi, range_min=-torch.pi, range_max=torch.pi)
        omega = modulo_with_wrapped_range(omega, range_min=-torch.pi, range_max=torch.pi)
        C_N_CA = modulo_with_wrapped_range(C_N_CA, range_min=-torch.pi, range_max=torch.pi)
        N_CA_C = modulo_with_wrapped_range(N_CA_C, range_min=-torch.pi, range_max=torch.pi)
        CA_C_N = modulo_with_wrapped_range(CA_C_N, range_min=-torch.pi, range_max=torch.pi)

    

    pred_coords = builder.cartesian_coords()
    pred_start = pred_coords[:, :3 ,:]
    pred_end = pred_coords[:, -3: ,:]
    pred_coords = builder.transform(pred_coords, pred_start, pred_end, fix_start, fix_end)
    TorchNERFBuilder.sv2pdb("./test.pdb", pred_coords[0])
    TorchNERFBuilder.sv2pdb("./raw.pdb", bb_coord[0].reshape(-1,3))
    print(builder.cartesian_coords())


if __name__ == "__main__":
    test_nerf()
    # test_gradient()
