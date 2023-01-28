import torch
import numpy as np 
import torch.nn as nn
import copy 

def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

def get_naswot(model, input, target, device):
    model = copy.deepcopy(model)
    model.eval()
    batch_size = input.shape[0]
    model.K = torch.zeros(batch_size, batch_size).cuda()

    def counting_forward_hook(module, inp, out):
        try:
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            if x.cpu().numpy().sum() == 0:
                model.K = model.K
            else:
                K2 = (1.-x) @ (1.-x.t())
                model.K = model.K + K + K2
        except:
            pass

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            module.register_forward_hook(counting_forward_hook)
    
    input = input.cuda(device=device)
    with torch.no_grad():
        model(input)
    score = hooklogdet(model.K.cpu().numpy())
    del model
    del input
    return score
