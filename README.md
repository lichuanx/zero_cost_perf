# Perf Neural Networks with Zero-cost Metrics at Initialization
This repo is amid to use zero-cost metrics to evluate neural netowrks at initialization, to provide basic features to predict model trained accuracy without any trainning process invoveld.

Supported Metrics:
- grad_norm
- snip
- grasp
- fisher
- jacob_cov
- plain
- synflow
- naswot
- zen
- ntk
- ssnr
- flops
- param_size

The details of all listed metrics please refer [Metrics_Documents](https://github.com/Tiaspetto/zero_cost_perf/blob/main/documents/metrics.md).

## Prepare Enviroments
1. Install Samsung Zero-cost-nas package, we have done some modify in this repo to support our new metrics ssnr

```
cd zero-cost-nas
pip intall .
```

2.  Install ptflops package

```
pip install ptflops
```

3. Install gpustat

```
pip install gpustat
```

## Example

```python
from netperf import get_model_perf_info
from dataloaders.dataloaders import define_dataloader
from utils import pick_gpu_lowest_memory
import torchvision.models as models

n_classes=10
device= pick_gpu_lowest_memory()
net = models.densenet161()
net.to(device)
loader = define_dataloader(dataset='CIFAR10', data='data', batch_size=64)

#default metrics will include['params', 'flops']
query_measures = ['zen', 'naswot', 'ssnr', 'synflow']
perf_dict = get_model_perf_info(net, loader, n_classes, device, measures=query_measures)
```
