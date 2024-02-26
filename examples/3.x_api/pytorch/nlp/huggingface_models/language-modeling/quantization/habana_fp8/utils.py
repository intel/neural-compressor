def show_msg():
    import numpy as np
    import glob
    from habana_frameworks.torch.hpu import memory_stats
    print("Number of HPU graphs:", len(glob.glob(".graph_dumps/*PreGraph*")))
    mem_stats = memory_stats()
    mem_dict = {
        "memory_allocated (GB)": np.round(mem_stats["InUse"] / 1024**3, 2),
        "max_memory_allocated (GB)": np.round(mem_stats["MaxInUse"] / 1024**3, 2),
        "total_memory_available (GB)": np.round(mem_stats["Limit"] / 1024**3, 2),
    }
    for k, v in mem_dict.items():
        print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))


def itrex_bootstrap_stderr(f, xs, iters):
    from lm_eval.metrics import _bootstrap_internal, sample_stddev
    res = []
    chunk_size = min(1000, iters)
    it = _bootstrap_internal(f, chunk_size)
    for i in range(iters // chunk_size):
        bootstrap = it((i, xs))
        res.extend(bootstrap)
    return sample_stddev(res)


def save_to_excel(dict):
    import pandas as pd
    df_new = pd.DataFrame(dict)
    try:
        df_existing = pd.read_excel('output.xlsx')
    except FileNotFoundError:
        df_existing = pd.DataFrame()
    df_combined = pd.concat([df_existing, df_new], axis=0, ignore_index=True)
    df_combined.to_excel('output.xlsx', index=False, engine='openpyxl', header=True)
