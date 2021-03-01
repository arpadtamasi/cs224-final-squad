def mask_logits(target, mask):
    return target * (1 - mask) + mask * (-1e30)
