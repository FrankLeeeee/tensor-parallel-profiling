import torch
import time
from model import LinearModel, LayerNormModel


def build_model(model_type, dim, checkpoint=True, mlp_ratio=4):
    if model_type == 'linear':
        return LinearModel(dim, mlp_ratio, checkpoint)
    elif model_type == 'layernorm':
        return LayerNormModel(dim, checkpoint=checkpoint)

def get_time_stamp():
    torch.cuda.synchronize()
    return time.time()

def get_memory_states():
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    max_cached = torch.cuda.max_memory_cached() / (1024 ** 3)
    torch.cuda.reset_peak_memory_stats()
    return max_allocated, max_cached