import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOCK_BITS=6
PAGE_BITS=6
HASH_BITS=16
WINDOW_SIZE=100000
LATENCY=0
PREFETCH_QUE_SIZE=10

BATCH_SIZE = 256
GAMMA = 0.999 # forget index
EPS_START = 0.95 #random>eps, use model, else random: start with highly random,
EPS_END = 0.001
EPS_DECAY = 100
MEMORY_SIZE=3000