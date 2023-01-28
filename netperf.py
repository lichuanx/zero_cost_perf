from ptflops import get_model_complexity_info
from zero_cost_metrics import score_network, ZC_COLLECTIONS

def get_model_perf_info(network, dataloader, clases, device, measures=[]):
    
    perf_dict = {}
    for m in measures:
        assert m in ZC_COLLECTIONS, "metrics is not in ZC_COLLECTIONS, please check"
        score = score_network(m, network,dataloader,clases, device=device)
        perf_dict[m] = score
    
    #sample a batch
    x, y = next(iter(dataloader))
    b, c, w, h = x.size()
    macs, params = get_model_complexity_info(network, (c, w, h), as_strings=False,
                                        print_per_layer_stat=False, verbose=False)
    
    perf_dict['flops'] = macs
    perf_dict['params'] = params 
    
    return perf_dict