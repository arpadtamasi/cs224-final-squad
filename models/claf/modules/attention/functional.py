def add_masked_value(tensor, mask, value=-1e7):
    mask = mask.float()
    values_to_add = value * mask
    return tensor * (1 - mask) + values_to_add
