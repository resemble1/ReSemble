import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
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

device = cf.device

#device="cpu"
BLOCK_BITS=cf.BLOCK_BITS
BATCH_SIZE = cf.BATCH_SIZE
GAMMA = cf.GAMMA # forget index
EPS_START = cf.EPS_START #random>eps, use model, else random: start with highly random,
EPS_END = cf.EPS_END
EPS_DECAY = cf.EPS_DECAY
MEMORY_SIZE=cf.MEMORY_SIZE
HIDDEN_LAYER=cf.HIDDEN_LAYER

steps_done = 0


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)    
    

class DQN(nn.Module):

    def __init__(self, input_d, outputs):
        super(DQN, self).__init__()
        self.input_d=input_d
        self.output_d=outputs
        self.fc1 = nn.Linear(input_d, HIDDEN_LAYER)  # 5*5 from image dimension
        #elf.fc2 = nn.Linear(10, 20)
        #self.fc3 = nn.Linear(20, 5)
        self.head = nn.Linear(HIDDEN_LAYER, outputs)


    def forward(self, x):
        x=x.view(-1,self.input_d)
        x = x.to(device)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        return self.head(x.view(x.size(0), -1))

def select_action(state,policy_net,n_actions):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
    #if sample < GAMMA:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            #return policy_net(state).max(1)[1].view(1, 1)
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(memory,policy_net):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
def rl_train(cache,memory,policy_net,target_net,model_save_path):
    pref_action_dict={"id":[],"action":[],"rl":[]}
    n_pref_list=[]
    n_pref=np.array([0,0,0,0,0])
    reward_list=[]
    num_episodes=30
    policy_optim=20
    target_update=1000
    
    cache.reset()#?cache.reset()#?
    reward_total=0
    print("miss_len:",cache.miss_len)
    for t in tqdm(range(cache.miss_len)):
        state=cache.state
        action = select_action(state,policy_net,cache.n_actions)
        n_pref[action.item()]+=1
        #new state
        reward_old=reward_total
        next_state, reward, done, curr_id, curr_pref_addr = cache.step(action.item())
        reward = torch.tensor([reward], device=device)
        pref_action_dict["id"].extend([curr_id])
        pref_action_dict["action"].extend([action.item()])
        pref_action_dict["rl"].extend([curr_pref_addr])
        
        memory.push(state,action,next_state,reward)
        
        if t % policy_optim == 0:
            optimize_model(memory,policy_net)
    
    # Update the target network, copying all weights and biases in DQN
        if t % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(target_net.state_dict(), model_save_path)
            reward_list.append(reward_total + 1)
            #n_pref_list.append(n_pref_norm)
            reward_total=0
        if done:
            target_net.load_state_dict(policy_net.state_dict())
            break    
    
    return pref_action_dict,target_net

def convert_blk_n_hex(pred_addr_blk):
    res=int(int(pred_addr_blk)<<(BLOCK_BITS))
    res2=res.to_bytes(((res.bit_length() + 7) // 8),"big").hex().lstrip('0')
    return res2

#%%
'''
WARM=2
TOTAL=3
path_cache="/home/pengmiao/Disk/work/HPCA/ML-DPC-S0/LoadTraces/spec17/623.xalancbmk-s0.txt.xz"
path_bo="/home/pengmiao/Disk/work/HPCA/pref_trace/ALL_2_8/spec17/623.xalancbmk-s0.trace.xz.bo_file.txt"
path_spp="/home/pengmiao/Disk/work/HPCA/pref_trace/ALL_2_8/spec17/623.xalancbmk-s0.trace.xz.spp_file.txt"
path_isb="/home/pengmiao/Disk/work/HPCA/pref_trace/ALL_2_8/spec17/623.xalancbmk-s0.trace.xz.sisb_file.txt"     
path_domino="/home/pengmiao/Disk/work/HPCA/pref_trace/ALL_2_8/spec17/623.xalancbmk-s0.trace.xz.domino_file.txt"

model_save_path="/home/pengmiao/Disk/work/HPCA/2_RL/results/623.xalancbmk-s0.rl_model.pth"
path_to_prefetch_file=model_save_path+".rl_20.csv"

path_spatial=[path_bo, path_spp]
path_temporal=[path_isb,path_domino]
'''
#%%
if __name__ == "__main__":   
    WARM=int(sys.argv[1])
    TOTAL=int(sys.argv[2])
    path_cache=sys.argv[3]
    path_spatial=sys.argv[4].split(";")
    path_temporal=sys.argv[5].split(";")
    model_save_path=sys.argv[6]
    path_to_prefetch_file=sys.argv[7]
    
    cache = Cache(path_cache, path_spatial, path_temporal, WARM,TOTAL)
    steps_done = 0
    memory = ReplayMemory(MEMORY_SIZE)
    policy_net = DQN(cache.n_state, cache.n_actions).to(device)
    target_net = DQN(cache.n_state, cache.n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters())
    
    pref_action_dict,target_net=rl_train(cache,memory,policy_net,target_net,model_save_path)
    
    df=pd.DataFrame(pref_action_dict)
    df["rl_hex"]=df.apply(lambda x: convert_blk_n_hex(x["rl"]),axis=1)
    #df.to_csv(path_to_prefetch_file,header=False, index=False, sep=" ")
    df[["id","action","rl_hex"]].to_csv(path_to_prefetch_file,header=0, index=False, sep=" ")
    torch.save(target_net.state_dict(), model_save_path)
    print ("saved")    

