import os
import torch
import json
from omegaconf import OmegaConf
from src.chroma.data import Protein
from model_interface import MInterface
from tqdm import tqdm
import ast


def batch_list(input_list, batch_size):
    """将输入列表分成指定大小的批次"""
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
    model.load_state_dict(checkpoint, strict=False)
    model = model.to('cuda')
    return model

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='/huyuqi/xmyu/FoldMLM/FoldMLM/datasets/EnzymeCommission/train', type=str)
    parser.add_argument('--save_vqid_path', default='./test.jsonl', type=str)
    parser.add_argument('--config', default='src/tools/foldtoken/model_zoom/FT4/config.yaml', type=str)
    # parser.add_argument('--checkpoint', default='./model_zoom/FT4/ckpt_consist.pth', type=str)
    parser.add_argument('--checkpoint', default='src/tools/foldtoken/model_zoom/FT4/ckpt.pth', type=str)
    parser.add_argument('--level', default=8, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model = load_model(args)

    file_name_list = ast.literal_eval(args.path_in)


    # # ================ reconstruct ============
    for file_list in tqdm(batch_list(file_name_list, 32)):
        list_of_protein = []
        list_of_title = []
        list_of_sequence = []
        for idx, file_name in enumerate(file_list):
            title = file_name.split('/')[-1].split('.')[0]
            input_pdb = file_name
            
            try:
                protein = Protein(input_pdb, device='cuda') 
                X,C,S = protein.to_XCS()
                sequence = protein.sequence(format="one-letter-string")
            except:
                continue
            if X.shape[1]<5:
                continue
            if (C!=-1).sum() < 5:
                continue
            list_of_protein.append(protein)
            list_of_title.append(title)
            list_of_sequence.append(sequence)

        try:
            torch.manual_seed(0)
            with torch.no_grad():
                batch = model.batch_proteins(list_of_protein)       
                list_protein_pred, list_vq_code, rmsd = model.sample(batch=batch,level=args.level, cal_rmsd=True)
        except:
            print(file_list)
            continue

        with open(args.save_vqid_path, 'a+') as file:
            batch_id = batch['batch_id']
            chain_encoding = batch['chain_encoding']
            S = batch['S']
            for idx, title in enumerate(list_of_title):
                # protein = list_protein_pred[idx]
                value = {
                    # 'seq': (S[batch_id==idx]).cpu().numpy().tolist(),
                    "seq": list_of_sequence[idx],
                    'vqid': (list_vq_code[idx]).cpu().numpy().tolist(),
                    # 'rmsd': rmsd[idx].cpu().numpy().item(),
                    # 'chain': chain_encoding[batch_id==idx].cpu().numpy().tolist()
                    }

                # 创建一个包含键值对的字典
                line_dict = {title: value}
                file.write(json.dumps(line_dict) + '\n')
            

