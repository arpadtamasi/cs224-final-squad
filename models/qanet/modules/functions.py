def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)
