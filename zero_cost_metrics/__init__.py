import torch
import numpy as np
from foresight.pruners import predictive
from zero_cost_metrics.naswot import get_naswot
from zero_cost_metrics.ntk import get_ntk_n
from zero_cost_metrics.zen import zennas_score
from zero_cost_metrics.naswot_layers import get_naswot_layerwise, get_naswot_perturbation_layerwise, get_naswot_mats_cache_layerwise
ZC_COLLECTIONS=[
    'grad_norm',
    'snip',
    'grasp',
    'fisher',
    'jacob_cov',
    'plain',
    'synflow',
    'naswot',
    'zen',
    'ntk',
    'ssnr',
    'nwot_ssnr',
    'nwot_ptb',
    'nwot_ptb_ssnr',
    'accum_nwot_ssnr',
]

def score_network(metric, model,train_queue,n_classes, device):
    if metric == 'naswot':
        input, target = next(iter(train_queue))
        return get_naswot(model, input, target, device)
    elif metric == 'ntk':
        ntk = get_ntk_n(train_queue, [model], device=device, recalbn=0, train_mode=True, num_batch=1)
        return ntk
    elif metric == 'zen':
        return zennas_score(model, device)
    elif metric == 'ssnr':
        measures = predictive.find_measures(model,
                                        train_queue,
                                        ('random', 1, n_classes), 
                                        device,
                                        measure_names=['synflow'], aggregate=False)
        measures = measures['synflow']
        #return measures
        scores = []
        for i in range(len(measures)):
            s = measures[i].detach().view(-1) 
            if torch.std(s) == 0:
                s = torch.sum(s)
            else:
                s = torch.sum(s)/torch.std(s)

            scores.append(s.cpu().numpy())
        return np.sum(scores)
    elif metric == 'nwot_ssnr':
        input, target = next(iter(train_queue))
        nwots = get_naswot_layerwise(model, input, target, device)
        measures = predictive.find_measures(model,
                                train_queue,
                                ('random', 1, n_classes), 
                                device,
                                measure_names=['synflow'], aggregate=False)
        measures = measures['synflow']
        #return measures
        scores = []
        for i in range(len(measures)):
            s = measures[i].detach().view(-1) 
            if torch.std(s) == 0:
                s = torch.sum(s)
            else:
                s = torch.sum(s)/torch.std(s)
            s = s * nwots[i]

            scores.append(s.cpu().numpy())
        return np.sum(scores)
    elif metric == 'nwot_ptb':
        input, target = next(iter(train_queue))
        nwots, accum_mats_logdet, accum_mats_logdetimp = get_naswot_perturbation_layerwise(model, input, target, device)
        return np.sum(accum_mats_logdetimp)
    elif metric == 'nwot_ptb_ssnr':
        input, target = next(iter(train_queue))
        nwots, accum_mats_logdet, accum_mats_logdetimp = get_naswot_perturbation_layerwise(model, input, target, device)
        measures = predictive.find_measures(model,
                                train_queue,
                                ('random', 1, n_classes), 
                                device,
                                measure_names=['synflow'], aggregate=False)
        measures = measures['synflow']
        #return measures
        scores = []
        for i in range(len(measures)):
            s = measures[i].detach().view(-1) 
            if torch.std(s) == 0:
                s = torch.sum(s)
            else:
                s = torch.sum(s)/torch.std(s)
            s = s * accum_mats_logdetimp[i]

            scores.append(s.cpu().numpy())
        return np.sum(scores)
    elif metric == 'accum_nwot_ssnr':
        input, target = next(iter(train_queue))
        K_mats, K_mats_logdet, K_accum_mats, K_accum_mats_logdet, accum_mats_logdetimp = get_naswot_mats_cache_layerwise(model, input, target, device)
        measures = predictive.find_measures(model,
                                train_queue,
                                ('random', 1, n_classes), 
                                device,
                                measure_names=['synflow'], aggregate=False)
        measures = measures['synflow']
        #return measures
        scores = []
        for i in range(len(measures)):
            s = measures[i].detach().view(-1) 
            if torch.std(s) == 0:
                s = torch.sum(s)
            else:
                s = torch.sum(s)/torch.std(s)
            s = s * accum_mats_logdetimp[i]

            scores.append(s.cpu().numpy())
        return np.sum(scores)
    else:
        measures = predictive.find_measures(model,
                                    train_queue,
                                    ('random', 1, n_classes), 
                                    device,
                                    measure_names=[metric], aggregate=True)
        
        return measures[metric]