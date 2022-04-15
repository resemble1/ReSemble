import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
import lzma
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config as cf

BLOCK_BITS=cf.BLOCK_BITS
PAGE_BITS=cf.PAGE_BITS
HASH_BITS=cf.HASH_BITS
WINDOW_SIZE=cf.WINDOW_SIZE
LATENCY=cf.LATENCY
PREFETCH_QUE_SIZE=cf.PREFETCH_QUE_SIZE

def read_load_trace_data(load_trace, num_prefetch_warmup_instructions=10,TOTAL_NUM=20):
    
    def process_line(line):
        split = line.strip().split(', ')
        return int(split[0]), int(split[1]), int(split[2], 16), int(split[3], 16), split[4] == '1'

    train_data = []
    eval_data = []
    if load_trace[-2:] == 'xz':
        with lzma.open(load_trace, 'rt') as f:
            for line in f:
                pline = process_line(line)
                if pline[0] < num_prefetch_warmup_instructions * 1000000:
                    train_data.append(pline)
                else:
                    eval_data.append(pline)
    else:
        with open(load_trace, 'r') as f:
            for line in f:
                pline = process_line(line)
                if pline[0] < num_prefetch_warmup_instructions * 1000000:
                    train_data.append(pline)
                else:
                    eval_data.append(pline)
    
    df=pd.DataFrame(eval_data)
    df.columns=["id", "cycle", "addr", "ip", "hit"]
    if TOTAL_NUM != None:
        df=df[df["id"]<TOTAL_NUM*1000000]
        
    df["addr_blk"]=[ x >> BLOCK_BITS for x in df['addr']]
    #return train_data, eval_data
    return df[["id", "addr_blk", "ip", "hit"]]

def read_prefetch_data(load_trace, num_prefetch_warmup_instructions=10,TOTAL_NUM=20):
    
    def process_line(line):
        split = line.strip().split(' ')
        return int(split[0]), int(split[1], 16)

    train_data = []
    eval_data = []
    if load_trace[-2:] == 'xz':
        with lzma.open(load_trace, 'rt') as f:
            for line in f:
                pline = process_line(line)
                if pline[0] < num_prefetch_warmup_instructions * 1000000:
                    train_data.append(pline)
                else:
                    eval_data.append(pline)
    else:
        with open(load_trace, 'r') as f:
            for line in f:
                pline = process_line(line)
                if pline[0] < num_prefetch_warmup_instructions * 1000000:
                    train_data.append(pline)
                else:
                    eval_data.append(pline)
    

    if len(eval_data)==0:
        df=pd.DataFrame([[0,0]])
        df.columns=["id", "addr_blk"]
        return df[["id","addr_blk"]]
    else:
        df=pd.DataFrame(eval_data)
        df.columns=["id", "addr"]
        if TOTAL_NUM != None:
            df=df[df["id"]<TOTAL_NUM*1000000]
        df["addr_blk"]=[ x >> BLOCK_BITS for x in df['addr']]
        return df[["id","addr_blk"]]


def addr_hash(x,norm=True):
    x=int(x)
    t = x^(x>>32); 
    result = (t^(t>>HASH_BITS)) & (2**HASH_BITS-1); 
    if norm==False:
        return result
    else:
        return result/(2**HASH_BITS)
def apply_state_spatial_norm(addr,pref,norm=True):
    if pref==0:
        return 0 
    elif norm==False:
        return addr_hash(pref-addr,norm)
    else:
        return (pref-addr)/(2**PAGE_BITS)

def apply_state_temporal_hash(addr,pref,norm=True):
    if pref==0:
        return 0
    else:
        return addr_hash(pref-addr,norm)
    

class Cache(object):
    def __init__(self, path_cache, path_bo, path_spp, path_isb,path_domino, warm_up=10,total=20, model_type="nn",context=True):#tab;nn
        super(Cache, self).__init__()
        self.model_type=model_type
        self.norm= True if model_type=="nn" else False
        self.pointer = 0
        self.state_pointer=0#<degree,e.g. degree=2, pp=0,1
        self.action_space=['bo','spp', "isb","domino","np"]
        self.n_actions = len(self.action_space)
        self.n_state = 5 if context==True else 4
        #[s_addr,s_ip,s_bo,s_pp,s_isb,s_domino]
        self.miss_trace=self._build_cache(path_cache, path_bo, path_spp, path_isb,path_domino,warm_up,total)
        self.miss_len=len(self.miss_trace)
        self.pref_que=[]
    
    def _build_cache(self,path_cache, path_bo, path_spp, path_isb,path_domino, warm_up,total):
        df_miss_trace=read_load_trace_data(path_cache, warm_up,total)
        df_bo=read_prefetch_data(path_bo, warm_up,total)
        df_bo.columns=["id","bo"]
        df_spp=read_prefetch_data(path_spp, warm_up,total)
        df_spp.columns=["id","spp"]
        df_isb=read_prefetch_data(path_isb, warm_up,total)
        df_isb.columns=["id","isb"]
        df_domino=read_prefetch_data(path_domino, warm_up,total)
        df_domino.columns=["id","domino"]
        
        
        pd_cac=pd.merge(df_miss_trace,df_bo,on="id",how="left")
        pd_cac=pd.merge(pd_cac,df_spp,on="id",how="left")
        pd_cac=pd.merge(pd_cac,df_isb,on="id",how="left")
        pd_cac=pd.merge(pd_cac,df_domino,on="id",how="left")

        
        pd_cac["bo"]=pd_cac["bo"].astype('Int64')
        pd_cac["spp"]=pd_cac["spp"].astype('Int64')
        pd_cac["isb"]=pd_cac["isb"].astype('Int64')
        pd_cac["domino"]=pd_cac["domino"].astype('Int64')
        
        df=pd_cac.fillna(0)
        
        df["s_bo"]=df.apply(lambda x: apply_state_spatial_norm(x["addr_blk"],x["bo"],self.norm),axis=1)
        df["s_spp"]=df.apply(lambda x: apply_state_spatial_norm(x["addr_blk"],x["spp"],self.norm),axis=1)
        df["s_isb"]=df.apply(lambda x: apply_state_temporal_hash(x["addr_blk"],x["isb"],self.norm),axis=1)
        df["s_domino"]=df.apply(lambda x: apply_state_temporal_hash(x["addr_blk"],x["isb"],self.norm),axis=1)
        if self.n_state == 5:
            #df["s_addr"]=df.apply(lambda x: addr_hash(x["addr_blk"],self.norm),axis=1)
            df["s_ip"]=df.apply(lambda x: addr_hash(x["ip"],self.norm),axis=1)
            df["state"]=df[['s_ip', 's_bo',"s_spp","s_isb","s_domino"]].values.tolist()
        else:
            df["state"]=df[['s_bo',"s_spp","s_isb","s_domino"]].values.tolist()
        return df
        

    def reset(self):
        self.pointer=0
        self.state_pointer=0
        pass
    
    @property
    def state(self):
        state_vec=self.miss_trace["state"][self.pointer:self.pointer+1].values[0]
        #state_t=torch.from_numpy(np.array(state_vec,dtype=np.int))
        return state_vec if self.model_type=="tab" else torch.from_numpy(np.array(state_vec,dtype=np.float32))
    
    @property
    def next_state(self):
        state_vec=self.miss_trace["state"][self.state_pointer:self.state_pointer+1].values[0]
        #state_t=torch.from_numpy(np.array(state_vec,dtype=np.float32))
        return state_vec if self.model_type=="tab" else torch.from_numpy(np.array(state_vec,dtype=np.float32))
    
    def get_reward(self,curr_pref_addr):
        future_window = self.miss_trace["addr_blk"][self.pointer+LATENCY:self.pointer+LATENCY+WINDOW_SIZE+1].values
        check_set=set(future_window)-set(self.pref_que)
        if curr_pref_addr in set(future_window):
            reward = 1
        else:
            reward = -1
        return reward
        
    
    def step(self,action):
        #todo: action->curr_pref_addr, curr_addr wrongly used now
        done=0
        curr_id=self.miss_trace["id"].values[self.pointer]
        #curr_addr=self.miss_trace["addr_blk"].values[self.pointer]
        
        if action == 0:
            pref ="bo"
        elif action == 1:
            pref = "spp"
        elif action == 2:
            pref = "isb"
        elif action == 3:
            pref = "domino"
        else:
            pref = "np"
        
        #pref=action
        if pref != "np":
            curr_pref_addr=self.miss_trace[pref].values[self.pointer]
            reward=self.get_reward(curr_pref_addr)
            #if curr_pref_addr>0:
            self.pref_que.append(curr_pref_addr)
            self.pref_que=self.pref_que[-PREFETCH_QUE_SIZE:]
        else:
            curr_pref_addr=0
            reward=0
        
        self.pointer+=1
        if self.pointer > (self.miss_len-1):
            self.reset()
            s_=self.state
            done=1
            reward=0
            return s_, reward, done, curr_id, curr_pref_addr
        else:
            new_id=self.miss_trace["id"].values[self.pointer]
            
            self.state_pointer=self.pointer+1
            if self.state_pointer > (self.miss_len-1):
                self.state_pointer=self.miss_len-1
            
            
            while curr_id == new_id:
                self.state_pointer=self.state_pointer+1
                new_id=self.miss_trace["id"].values[self.state_pointer]
                if self.state_pointer > (self.miss_len-1):
                    self.state_pointer=self.miss_len-1
                    break
                
            s_=self.next_state
            #curr_pref_addr: blk addr, not final result    
            return s_, reward, done, curr_id, curr_pref_addr