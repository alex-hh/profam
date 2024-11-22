import inspect
from torch.utils.data import DataLoader
from src.interface.data_interface import DInterface_base
import torch

class MyDataLoader(DataLoader):
    def __init__(self, data_module, dataset, num_workers=8, virtual_frame_num=0, *args, **kwargs):
        super(MyDataLoader, self).__init__(dataset, num_workers=num_workers, *args, **kwargs)
        self.data_module = data_module
        self.pretrain_device = 'cuda:0'
        self.virtual_frame_num = virtual_frame_num
       
    
    def __iter__(self):
        for batch in super().__iter__():
            # 在这里对batch进行处理
            # ...
            try:
                self.pretrain_device = f'cuda:{torch.distributed.get_rank()}'
            except:
                self.pretrain_device = 'cuda:0'
            yield batch


def memory_efficient_collate_fn(batch):
    batch = [one for one in batch if one is not None]
    batch = [one for one in batch if len(one['X'])>0]
    if len(batch)==0:
        return None
    num_nodes = torch.tensor([one['num_nodes'] for one in batch])
    shift = num_nodes.cumsum(dim=0)
    shift = torch.cat([torch.tensor([0], device=shift.device), shift], dim=0)

    
    ret = {}
    for key in batch[0].keys():
        if key in ['edge_idx']:
            ret[key] = torch.cat([one[key] + shift[idx] for idx, one in enumerate(batch)], dim=1)
        elif key in ['batch_id']:
            ret[key] = torch.cat([one[key] + idx for idx, one in enumerate(batch)])
        elif type(batch[0][key])==torch.Tensor:
            ret[key] = torch.cat([one[key] for one in batch], dim=0)
        elif type(batch[0][key])== str:
            ret[key] = [one[key] for one in batch]

    return ret



class DInterface(DInterface_base):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.load_data_module()
        self.dataset = self.instancialize(split = 'train')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(split = 'train', data=self.dataset.all_data['train'])

            self.valset = self.instancialize(split = 'val', data=self.dataset.all_data['val'])

        if stage == 'test' or stage is None:
            self.testset = self.instancialize(split = 'test', data=self.dataset.all_data['test'])

    def train_dataloader(self):
        # self.trainset.dynamic_mix_AF2DB()
        return MyDataLoader(self, self.trainset,  batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,  prefetch_factor=3, pin_memory=True, collate_fn=memory_efficient_collate_fn, virtual_frame_num = self.hparams.virtual_frame_num)

    def val_dataloader(self):
        return MyDataLoader(self, self.valset,  batch_size=self.hparams.batch_size,  num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=memory_efficient_collate_fn, virtual_frame_num = self.hparams.virtual_frame_num, prefetch_factor=3)

    def test_dataloader(self):
        return MyDataLoader(self, self.testset,  batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=memory_efficient_collate_fn, virtual_frame_num = self.hparams.virtual_frame_num,   prefetch_factor=3)

    def load_data_module(self):
        name = self.hparams.dataset
        name = self.dataset
        
        if name == 'CATH_dataset':
            from src.datasets.cath_dataset_struct import CATHDataset
            self.data_module = CATHDataset
        
        if name == 'PDB_dataset':
            from src.datasets.pdb_dataset_struct import PDBDataset
            self.data_module = PDBDataset

        if name == 'AF2DB_dataset':
            from src.datasets.pdb_dataset_struct import PDBDataset
            self.data_module = PDBDataset

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        
        class_args =  list(inspect.signature(self.data_module.__init__).parameters)[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.hparams[arg]
        args1.update(other_args)
        return self.data_module(**args1)