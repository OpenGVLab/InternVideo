import torch
import random
import numpy as np

def set_seed(seed, rank, world_size):
    rng = random.Random(seed)
    seed_per_rank = [rng.randint(0, 2**32-1) for _ in range(world_size)]
    cur_seed = seed_per_rank[rank]
    random.seed(cur_seed)
    torch.manual_seed(cur_seed)
    torch.cuda.manual_seed(cur_seed)
    np.random.seed(cur_seed)