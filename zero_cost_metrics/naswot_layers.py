import torch
import numpy as np 
import torch.nn as nn
import copy 
import traceback


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
                model.K_accum_mats.append(matrix if (len(model.K_accum_mats) == 0) else (model.K_accum_mats[-1]+matrix))
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
    accum_mats = model.K_accum_mats
    accum_mats_logdet = model.K_accum_mats_logdet
    # magical calc here||||
    a_ori = accum_mats_logdet.copy() 
    a_before = a_ori[:-1]
    a_after = a_ori[1:]
    a_diff = [i-j for (i,j) in zip(a_after, a_before)] # calc subtraction between neighbour elements.
    a_diff.insert(0, 0) # ensure all models' first layer has the same default importance of 0.
    accum_mats_logdet_imp = a_diff
    # # print(scores, accum_mats, accum_mats_logdet, a_diff)
    # print('-'*20)
    # print(scores)
    # print('-'*20)
    # print(accum_mats_logdet)
    # print('-'*20)
    # print(a_diff)
    # print('-'*20)
    del model
    del input
    # return scores, accum_mats, accum_mats_logdet
    # return scores, None, accum_mats_logdet, accum_mats_logdet_imp
    return scores, accum_mats_logdet, accum_mats_logdet_imp


def get_naswot_mats_cache_layerwise(model, input, target, device) -> tuple[list, list, list, list, list]:
    '''
    get and store all cached intermediate mats:
        - `naswot layerwise mats` (per layer per mat)
        - - `naswot original` BIG matrix, element-wise sum all `naswot layerwise mats`
        - - `naswot accumulated` matrix list, accumulate `naswot layerwise mats` layerly, 
    '''
    model = copy.deepcopy(model)
    model.eval()
    batch_size = input.shape[0]  # TODO: we won't guarantee this first element is batchsize, since it would change in NASBENCH101-micro
    # #model.K = torch.zeros(batch_size, batch_size).cuda()
    # model.K_dict = {}  # dict of **naswot matrix logdet**, layer-wise | <k,v>(<model_name,layer_logdet>
    zeros_mat = torch.zeros(batch_size, batch_size).to(device=device)
    K_layer_names = [] # list of registered layer (module) names.
    K_mats = [] # list of **naswot matrix**, layer-wise | [e]([mat, ...])
    K_mats_logdet = [] # list of **naswot matrix**, layer-wise | [e]([mat, ...])
    K_accum_mats = [] # list of **accumulated naswot matrix**, layer-wise | [e]([accum_matrix, ...])
    K_accum_mats_logdet = [] # list of **logdet of accumulated naswot matrix**, layer-wise | [e]([accum_matrix_logdet, ...])

    def counting_forward_hook(module, inp, out):
        # try:
        out = out.view(out.size(0), -1)
        x = (out > 0).float()
        K = x @ x.t()
        K2 = (1.-x) @ (1.-x.t())
        matrix = K + K2

        K_layer_names.append(module.alias)
        K_mats.append(matrix)
        K_mats_logdet.append(safe_hooklogdet(K_mats[-1].cpu().numpy()))
        K_accum_mats.append(matrix if (len(K_accum_mats) == 0) else (K_accum_mats[-1]+matrix))
        K_accum_mats_logdet.append(safe_hooklogdet(K_accum_mats[-1].cpu().numpy()))
        # except Exception:
        #     print('counting_forward_hook ... error')
        #     print(traceback.format_exc())

    # register forward hook fn
    registered_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            module.alias = name
            module.register_forward_hook(counting_forward_hook)
            registered_layers.append(name)
    
    # input = input.cuda(device=device)
    input = input.to(device=device)
    with torch.no_grad():
        model(input)
    scores = []

    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
    #         scores.append(model.K_dict[name])
    assert registered_layers == K_layer_names, 'Not all module forward hook fn were triggered successfully'
    #scores = copy.deepcopy(scores)
    
    # magical calc here||||
    # a_ori = K_accum_mats_logdet.copy() 
    a_before = K_accum_mats_logdet[:-1]
    a_after = K_accum_mats_logdet[1:]
    a_diff = [i-j for (i,j) in zip(a_after, a_before)] # calc subtraction between neighbour elements.
    # a_diff.insert(0, 0) # ensure all models' first layer has the same default importance of 0.
    a_diff.insert(0, K_accum_mats_logdet[0]) # ensure all models' first layer has the same default importance of 0.
    scores = a_diff
    
    del model
    del input
    # return scores, accum_mats, accum_mats_logdet
    # return scores, None, accum_mats_logdet, accum_mats_logdet_imp
    return K_mats, K_mats_logdet, K_accum_mats, K_accum_mats_logdet, scores
