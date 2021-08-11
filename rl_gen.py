import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
#from PIL import Image
import lzma
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from cache_env import Cache,addr_hash
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOCK_BITS=6
BATCH_SIZE=256


class MAPDataset(Dataset):
    def __init__(self, df):
        self.df_state=list(df["state"].values)

    def __getitem__(self, idx):
        
        state = self.df_state[idx]
        return state

    def __len__(self):
        return len(self.df_state)
    
    
    def collate_fn(self, batch):      
        state_b_tensor=torch.FloatTensor(batch).to(device)
        return state_b_tensor
    
class DQN(nn.Module):

    def __init__(self, input_d, outputs):
        super(DQN, self).__init__()
        self.input_d=input_d
        self.output_d=outputs
        self.fc1 = nn.Linear(input_d, 10)  # 5*5 from image dimension
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 5)
        self.head = nn.Linear(5, outputs)


    def forward(self, x):
        x=x.view(-1,self.input_d)
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x.view(x.size(0), -1))

def select_action_gen(state):
    return target_net(state).max(1)[1]

def apply_state_bo(addr,bo):
    return (bo-addr)/64

def apply_state_delta_hash(addr,pref):
    return addr_hash(pref-addr)

def pick_addr(bo,spp,isb,domino,pick):
    if pick == 0:
        return bo
    elif pick == 1:
        return spp
    elif pick == 2:
        return isb
    elif pick == 3:
        return domino
    else:
        return 0
    
def convert_blk_n_hex(pred_addr_blk):
    res=int(int(pred_addr_blk)<<(BLOCK_BITS))
    res2=res.to_bytes(((res.bit_length() + 7) // 8),"big").hex().lstrip('0')
    return res2

def output_partition(df,path):
    pick_dict={}
    pick_list=df["pick"].values.tolist()
    pick_dict["BO"]=[pick_list.count(0)]
    pick_dict["SPP"]=[pick_list.count(1)]
    pick_dict["ISB"]=[pick_list.count(2)]
    pick_dict["Domino"]=[pick_list.count(3)]
    pick_dict["NP"]=[pick_list.count(4)]
    df_pick=pd.DataFrame(pick_dict)
    df_pick.to_csv(path,header=True,index=False)
    return df_pick

def output_pref_file(df,item,path):
    df["pref_hex"]=df.apply(lambda x: convert_blk_n_hex(x[item]),axis=1)
    df2=df[df["pref_hex"]!=""]
    df2[["id","pref_hex"]].to_csv(path,header=False, index=False, sep=" ")
    return

#%%
'''
WARM=3
TOTAL=4
path_cache="/home/pengmiao/Disk/work/HPCA/ML-DPC-S0/LoadTraces/spec17/623.xalancbmk-s0.txt.xz"
path_bo="/home/pengmiao/Disk/work/HPCA/pref_trace/ALL_2_8/spec17/623.xalancbmk-s0.trace.xz.bo_file.txt"
path_spp="/home/pengmiao/Disk/work/HPCA/pref_trace/ALL_2_8/spec17/623.xalancbmk-s0.trace.xz.spp_file.txt"
path_isb="/home/pengmiao/Disk/work/HPCA/pref_trace/ALL_2_8/spec17/623.xalancbmk-s0.trace.xz.sisb_file.txt"     
path_domino="/home/pengmiao/Disk/work/HPCA/pref_trace/ALL_2_8/spec17/623.xalancbmk-s0.trace.xz.domino_file.txt"

model_save_path="/home/pengmiao/Disk/work/HPCA/2_RL/results/spec17/623.xalancbmk-s0.trace.xz.pth"
path_prefetch_file_root = "/home/pengmiao/Disk/work/HPCA/2_RL/results/gen_pref_files/623.xalancbmk-s0.trace.xz."
path_partition="/home/pengmiao/Disk/work/HPCA/2_RL/results/partition/623.xalancbmk-s0.trace.xz."
'''
#%%
if __name__ == "__main__":   
    cache = Cache(path_cache, path_bo, path_spp, path_isb,path_domino, WARM,TOTAL)
    n_actions = cache.n_actions
    n_state=cache.n_state
    target_net = DQN(n_state, n_actions)
    target_net.load_state_dict(torch.load(model_save_path))    
    target_net.to(device)
    target_net.eval()
    df=cache.miss_trace
    
    test_dataset=MAPDataset(df)
    test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=test_dataset.collate_fn)
    
    prediction=[]
    for data in tqdm(test_loader):
        output= select_action_gen(data)
        prediction.extend(output.cpu().tolist())
    
    df["pick"]= prediction
    df["rl"]=df.apply(lambda x: pick_addr(x["bo"],x["spp"],x["isb"],x["domino"],x["pick"]),axis=1)    
    
    # 80 #
    # pref_file*5: rl, bo, spp, isb, domino
    # parition_file*1
    items = ["rl","bo","spp","isb","domino"]
    for item in items:
        output_pref_file(df,item,path_prefetch_file_root+item+".pref.80.csv")
    
    output_partition(df,path_partition+"partition.80.csv")
    
    # 100 #
    file=model_save_path+".rl_train.csv"
    df_train=pd.read_csv(file,sep=" ")
    df_train.columns=(["id","pick","rl","rl_hex"])
    df_all=pd.concat([df_train[["id","pick","rl"]], df[["id","pick","rl"]]], axis=0)
    item="rl"
    output_pref_file(df_all,item,path_prefetch_file_root+item+".pref.100.csv")
    output_partition(df_all,path_partition+"partition.100.csv")


#%%

'''
DONOT DELETE!!
df_1=pd.read_csv(path_partition,header=0)
df_2=pd.read_csv(path_partition)
df_conc = pd.concat( [df_1, df_2], axis=0)
'''




