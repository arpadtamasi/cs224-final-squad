import torch

def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)

def recurse(func, initial, max):
    value = initial
    for _ in range(max):
        value = func(value)
        yield value
