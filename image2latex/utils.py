import torch
import numpy as np


def exact_match(truth, pred):
    len_truth = len(truth)
    len_pred = len(pred)
    max_len = max(len_truth, len_pred)
    a = truth + [""] * (max_len - len_truth)
    b = pred + [""] * (max_len - len_pred)
    em = np.mean(np.array(a) == np.array(b))
    return torch.tensor(em)
