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
from cache_env import Cache
from tqdm import tqdm
import sys
import os
import config as cf
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALPHA = cf.ALPHA
BLOCK_BITS=cf.BLOCK_BITS
GAMMA=cf.GAMMA
EPS_START = cf.EPS_START #random>eps, use model, else random: start with highly random,
EPS_END = cf.EPS_END
EPS_DECAY = cf.EPS_DECAY
EPISOD=cf.EPISOD
ACTION=cf.ACTION
HASH_BITS=cf.HASH_BITS

steps_done = 0

#%%

def select_action(state,q_table):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if state in q_table:
        state_actions = q_table[state]
    else:
        state_actions=[0,0,0,0,0]
        q_table[state]=state_actions
    if ((sample < eps_threshold) or (not any(state_actions))):
        action_name = np.random.choice(range(len(ACTION)))
    else:
        action_name = state_actions.index(max(state_actions))#idxmax: the index of max value
    return action_name, q_table

def convert_blk_n_hex(pred_addr_blk):
    res=int(int(pred_addr_blk)<<(BLOCK_BITS))
    res2=res.to_bytes(((res.bit_length() + 7) // 8),"big").hex().lstrip('0')
    return res2

def rl_train(cache):
    q_table={}
    pref_action_dict={"id":[],"action":[],"rl":[]}
    n_pref_list=[]
    reward_list=[]
    cache.reset()#?cache.reset()#?
    reward_total=0
    import pdb
    #pdb.set_trace()
    MAX_EPISODES=1
    for episode in range(MAX_EPISODES):
            cache.reset()
            n_pref=np.array([0,0,0,0,0])
            for t in tqdm(range(cache.miss_len)):
                state=tuple(cache.state)
                action, q_table = select_action(state,q_table)
                #action=1
                #new stateo
                q_predict = q_table[state][action]
                n_pref[action]+=1
                
                next_state, reward, done, curr_id, curr_pref_addr = cache.step(action)
                reward_total+=reward
                next_state=tuple(next_state)
                if next_state in q_table:
                    state_actions = q_table[next_state]
                else:
                    state_actions=[0,0,0,0,0]
                    q_table[next_state]=state_actions
                
                if done != 1:
                    q_target = reward + GAMMA * max(q_table[next_state])  # next state is not terminal
                else:
                    q_target = reward     # next state is terminal
                    done = 1    # terminate this episode
                
                q_table[state][action] += ALPHA * (q_target - q_predict)  # update
                
                
                pref_action_dict["id"].extend([curr_id])
                pref_action_dict["action"].extend([action])
                pref_action_dict["rl"].extend([curr_pref_addr])
                if t%EPISOD==0:
                    if t==0:
                        n_pref=np.array([0.2,0.2,0.2,0.2,0.2])
                    reward_list.append(reward_total)
                    n_pref_list.append((n_pref/sum(n_pref)).copy())
                    n_pref=np.array([0,0,0,0,0])
                    reward_total=0
    return reward_list,n_pref_list, pref_action_dict, q_table


#%%

if __name__ == "__main__":   
    
    WARM=int(sys.argv[1])
    TOTAL=int(sys.argv[2])
    
    path_cache=sys.argv[3]
    path_bo=sys.argv[4]
    path_spp=sys.argv[5]
    path_isb=sys.argv[6]
    path_domino=sys.argv[7]
    model_save_path=sys.argv[8]
    path_to_prefetch_file=sys.argv[9]
    path_to_stats=sys.argv[10]
    
    print("ReSemble: Tabular Variant")
    print("Hashing bit:", HASH_BITS)
    cache = Cache(path_cache, path_bo, path_spp, path_isb,path_domino, WARM,TOTAL,model_type="tab",context=False)
    print("Online updating...")
    reward_list,n_pref_list,pref_action_dict,q_table=rl_train(cache)

    df_list=pd.DataFrame(n_pref_list,columns=cf.ACTION)
    df_list["reward"]=reward_list
    df_list["Steps"]=[i*EPISOD for i in range(len(n_pref_list))]
    df_list.to_csv(path_to_stats,header=1, index=False,sep=",")
    print ("Rewards and proportions saved at:",path_to_stats)

    df=pd.DataFrame(pref_action_dict)
    df["rl_hex"]=df.apply(lambda x: convert_blk_n_hex(x["rl"]),axis=1)
    df_pref=df[["id","rl_hex"]]
    df_pref=df_pref[df_pref["rl_hex"]!=""]
    df_pref.to_csv(path_to_prefetch_file,header=False, index=False, sep=" ")
    print("Prefetching file generated at:", path_to_prefetch_file)
    #torch.save(target_net.state_dict(), model_save_path)
    with open(model_save_path, 'wb') as f:
        pickle.dump(q_table,f)
    print ("Model saved at:", model_save_path)

