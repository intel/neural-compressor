import numpy as np
import torch

def collate_torch_preds(results):
    batch = results[0]
    if isinstance(batch, list):
        results = zip(*results)
        collate_results = []
        for output in results:
           output = [batch.numpy() if isinstance(batch, torch.Tensor) else batch for batch in output]
           collate_results.append(np.concatenate(output))
    elif isinstance(batch, torch.Tensor):
        results = [batch.numpy() if isinstance(batch, torch.Tensor) else batch for batch in results]
        collate_results = np.concatenate(results)
    return collate_results
