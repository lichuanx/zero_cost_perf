import torch
import numpy as np
from foresight.pruners import predictive
from zero_cost_metrics.naswot import get_naswot
from zero_cost_metrics.ntk import get_ntk_n
from zero_cost_metrics.zen import zennas_score

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
    'ssnr'
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
    else:
        measures = predictive.find_measures(model,
                                    train_queue,
                                    ('random', 1, n_classes), 
                                    device,
                                    measure_names=[metric], aggregate=True)
        
        return measures[metric]