import torch
import numpy as np 
import torch.nn as nn
import copy 

def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld


def safe_hooklogdet(K):
    s, ld = np.linalg.slogdet(K)
    return 0 if (np.isneginf(ld) and s==0) else ld


def relative_entropy(batch):
    """
    Calculates the relative entropy within a batch of binary codes.

    Args:
    - batch: a 2D numpy array of shape (num_samples, num_bits) containing the binary codes

    Returns:
    - relative_entropy: a scalar value representing the relative entropy within the batch
    """
    # Convert batch to a numpy array of integers
    batch_int = batch.dot(1 << np.arange(batch.shape[1] - 1, -1, -1))

    # Count the number of occurrences of each integer in the batch
    counts = np.bincount(batch_int)

    # Compute the probability distribution of the integers
    probs = counts / float(batch.shape[0])

    # Calculate the entropy of the probability distribution
    entropy = -np.sum(probs * np.log2(probs))

    # Calculate the maximum possible entropy for a binary code of the same length
    max_entropy = batch.shape[1]

    # Calculate the relative entropy as the ratio of the actual entropy to the maximum entropy
    relative_entropy = entropy / max_entropy

    return relative_entropy

def get_bentropy_layerwise(model, input, target, device):
    model = copy.deepcopy(model)
    model.eval()
    batch_size = input.shape[0]
    #model.K = torch.zeros(batch_size, batch_size).cuda()
    model.entropy_dict = {}

    def counting_forward_hook(module, inp, out):
        try:
            out = out.view(out.size(0), -1)
            x = (out > 0).int()
            model.entropy_dict[module.alias] = relative_entropy(x)
        except:
            pass


    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            module.alias = name
            module.register_forward_hook(counting_forward_hook)
    
    # input = input.cuda(device=device)
    input = input.to(device=device)
    with torch.no_grad():
        model(input)
    scores = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            scores.append(model.K_dict[name])
    #scores = copy.deepcopy(scores)
    del model
    del input
    return scores