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


def get_naswot_layerwise(model, input, target, device):
    model = copy.deepcopy(model)
    model.eval()
    batch_size = input.shape[0]
    #model.K = torch.zeros(batch_size, batch_size).cuda()
    model.K_dict = {}

    def counting_forward_hook(module, inp, out):
        try:
            out = out.view(out.size(0), -1)
            x = (out > 0).float()
            K = x @ x.t()
            if x.cpu().numpy().sum() == 0:
                # model.K_dict[module.name] = 0
                model.K_dict[module.alias] = 0
            else:
                K2 = (1.-x) @ (1.-x.t())
                matrix = K + K2
                # model.K_dict[module.name] = hooklogdet(matrix.cpu().numpy())
                abslogdet = hooklogdet(matrix.cpu().numpy())
                model.K_dict[module.alias] = 0. if np.isneginf(abslogdet) else abslogdet #TODO: -inf
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


def get_naswot_perturbation_layerwise(model, input, target, device) -> tuple[list, list, list]:
    '''
    get accumulated importance through each layer.
    '''
    model = copy.deepcopy(model)
    model.eval()
    batch_size = input.shape[0]
    #model.K = torch.zeros(batch_size, batch_size).cuda()
    model.K_dict = {}  # dict of **naswot matrix logdet**, layer-wise | <k,v>(<model_name,layer_logdet>
    zeros_mat = torch.zeros(batch_size, batch_size).to(device=device)
    model.K_accum_mats = [] # list of **accumulated naswot matrix**, layer-wise | [e]([accum_matrix,...])
    model.K_accum_mats_logdet = [] # list of **logdet of accumulated naswot matrix**, layer-wise | [e]([accum_matrix_logdet,...])

    def counting_forward_hook(module, inp, out):
        try:
            out = out.view(out.size(0), -1)
            x = (out > 0).float()
            K = x @ x.t()
            if x.cpu().numpy().sum() == 0:
                # model.K_dict[module.name] = 0
                model.K_dict[module.alias] = 0
                model.K_accum_mats.append(zeros_mat if (len(model.K_accum_mats) == 0) else model.K_accum_mats[-1])
            else:
                K2 = (1.-x) @ (1.-x.t())
                matrix = K + K2
                # model.K_dict[module.name] = hooklogdet(matrix.cpu().numpy())
                abslogdet = hooklogdet(matrix.cpu().numpy())
                model.K_dict[module.alias] = 0. if np.isneginf(abslogdet) else abslogdet #TODO: -inf
                model.K_accum_mats.append(zeros_mat if (len(model.K_accum_mats) == 0) else (model.K_accum_mats[-1]+matrix))
            model.K_accum_mats_logdet.append(safe_hooklogdet(model.K_accum_mats[-1].cpu().numpy()))
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
    return scores, model.K_accum_mats, model.K_accum_mats_logdet
