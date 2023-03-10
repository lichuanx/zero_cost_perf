def pick_gpu_lowest_memory():
    import gpustat
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.memory_used)/float(gpu.memory_total), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU
