import os
import torch
from omegaconf import OmegaConf
from src.chroma.data import Protein
from model_interface import MInterface
import json

def batch_list(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]
    
def load_model(args):
    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)
    model = MInterface(**config)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cuda'))
    for key in list(checkpoint.keys()):
        if '_forward_module.' in key:
            checkpoint[key.replace('_forward_module.', '')] = checkpoint[key]
            del checkpoint[key]
    model.load_state_dict(checkpoint)
    model = model.to('cuda')
    return model

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='./N128', type=str)
    parser.add_argument('--path_out', default='./N128_pred', type=str)
    parser.add_argument('--config', default='./foldtoken/model_zoom/FT4/config.yaml', type=str)
    parser.add_argument('--checkpoint', default='./foldtoken/model_zoom/FT4/ckpt.pth', type=str)
    parser.add_argument('--level', default=8, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model = load_model(args)
    
    file_names = os.listdir(args.path_in)
    args.path_out = args.path_out+f'_level{args.level}/'
    os.makedirs(args.path_out, exist_ok=True)

    # # ================ reconstruct ============
    
    vqids = {}
    for file_list in batch_list(file_names, 32):
        list_of_protein = []
        list_of_title = []
        for idx, file_name in enumerate(file_list):
            title = file_name.split('.')[0]
            input_pdb = os.path.join(args.path_in, file_name)
            
            protein = Protein(input_pdb, device='cuda') 
            list_of_protein.append(protein)
            list_of_title.append(title)

    
        torch.manual_seed(0)
        with torch.no_grad():
            batch = model.batch_proteins(list_of_protein)       
            list_protein_pred, list_vq_code = model.sample(batch=batch,level=args.level)

        for idx, title in enumerate(list_of_title):
            if os.path.exists(input_pdb) == False:
                print(f'{input_pdb} does not exist, skip')
                continue
            output_prefix = args.path_out + title+'_pred.pdb'
            if os.path.exists(output_prefix):
                print(f'{output_prefix} exists, skip')
                continue
            
            list_protein_pred[idx].to(output_prefix, mask_indices=None, seq=list_vq_code[idx].tolist())
            
            vqids[title] = list_vq_code[idx].tolist()
    
    with open(args.path_out+'vqids.json', 'w') as f:
        json.dump(vqids, f)