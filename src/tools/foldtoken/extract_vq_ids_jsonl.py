import os
import torch
import json
from omegaconf import OmegaConf
from src.chroma.data import Protein
from model_interface import MInterface
from tqdm import tqdm

def safe_index(s):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return [alphabet.index(c) if c in alphabet else 0 for c in s]

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
    model.load_state_dict(checkpoint)
    model = model.to('cuda')
    return model

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='/huyuqi/xmyu/FoldToken2/foldtoken2_data/pdb/pdb.jsonl', type=str)
    parser.add_argument('--save_vqid_path', default='./pdb_vqid12.jsonl', type=str)
    parser.add_argument('--config', default='./model_zoom/FT4/config.yaml', type=str)
    parser.add_argument('--checkpoint', default='./model_zoom/FT4/ckpt.pth', type=str)
    parser.add_argument('--level', default=8, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model = load_model(args)
    

    with open(args.path_in, 'r') as f:
        lines = f.readlines()
    
    
    data_all = []
    for line in tqdm(lines):
        data = json.loads(line)
        X = torch.stack([torch.tensor(data['coords']['N']), 
                        torch.tensor(data['coords']['CA']), 
                        torch.tensor(data['coords']['C']), 
                        torch.tensor(data['coords']['O'])], dim=1)
        S = torch.tensor(safe_index(data['seq']))
        start_idx = 0
        
        ## ========== multi-chain ============
        name = data['name'].split('.')[0]
        C= []
        for cid, (cname, length) in enumerate(data['chain_length'].items()):
            C += [cid+1]*length
        C = torch.tensor(C)
        data_all.append((name, X, C, S))
        
        # ## ========== single-chain ============
        # for cid, (cname, length) in enumerate(data['chain_length'].items()):
        #     name = data['name'].split('.')[0] + f'_{cid}'
        #     C = torch.tensor([cid]*length)
        #     if length<30:
        #         continue
        #     data_all.append((name, X[start_idx:start_idx+length], C, S[start_idx:start_idx+length]))
        #     start_idx += length
        
        if len(data_all)>1000:
            break
        
    all_data = []
    # # ================ reconstruct ============
    for list_of_data in tqdm(batch_list(data_all, 32)):
        torch.manual_seed(0)
        with torch.no_grad():
            try:
                batch = model.batch_data_single_chain(list_of_data)       
                list_vq_code = model.encode_only(batch=batch,level=args.level)
            except:
                print(batch['title'])
                continue
        
        batch_id = batch['batch_id']
        chain_encoding = batch['chain_encoding']
        S = batch['S']
        for idx, title in enumerate(batch['title']):
            value = {'seq': (S[batch_id==idx]).cpu().numpy().tolist(),
                'vqid': (list_vq_code[idx]).cpu().numpy().tolist()}

            # 创建一个包含键值对的字典
            all_data.append({title: value})

        

    with open(args.save_vqid_path, 'a+') as file:
        for entry in all_data:
            file.write(json.dumps(entry) + '\n')
            

        


        



    