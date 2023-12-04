import neural_compressor as inc
print("neural_compressor version {}".format(inc.__version__))

import torch
print("torch {}".format(torch.__version__))

from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig, AccuracyCriterion, TuningCriterion, Options
from neural_compressor.data import DataLoader

import fashion_mnist
import alexnet


def ver2int(ver):
    s_vers = ver.split(".")
    res = 0
    for i, s in enumerate(s_vers):
        res += int(s)*(100**(2-i))

    return res

def compare_ver(src, dst):
    src_ver = ver2int(src)
    dst_ver = ver2int(dst)
    if src_ver>dst_ver:
        return 1
    if src_ver<dst_ver:
        return -1
    return 0

def auto_tune(input_graph_path, batch_size):
    train_dataset, test_dataset = fashion_mnist.download_dataset()
    model = alexnet.load_mod(input_graph_path)
    calib_dataloader = DataLoader(framework='pytorch', dataset=train_dataset, batch_size = batch_size)
    eval_dataloader = DataLoader(framework='pytorch', dataset=test_dataset, batch_size = batch_size)
    tuning_criterion = TuningCriterion(max_trials=100)
    option = Options(workspace='../output/nc_workspace')
    config = PostTrainingQuantConfig(approach="static", tuning_criterion=tuning_criterion,
    accuracy_criterion = AccuracyCriterion(
      higher_is_better=True, 
      criterion='relative',  
      tolerable_loss=0.01  
      )
    )
    q_model = fit(
        model=model,
        conf=config,
        calib_dataloader=calib_dataloader,
        eval_dataloader=eval_dataloader
        )

    return q_model

def main():
    batch_size = 200
    fp32_model_file = "../output/alexnet_mnist_fp32_mod.pth"
    int8_model = "../output/alexnet_mnist_int8_mod"
    
    if compare_ver(inc.__version__, "2.0")>=0:
        print(f"Compatible Intel Neural Compressor version detected : v{inc.__version__} ")
    else:
        raise Exception(f"Installed Intel Neural Compressor version[v{inc.__version__}] is NOT compatible. Please upgrade to version 2.0 or higher.")

    q_model = auto_tune(fp32_model_file, batch_size)
    q_model.save(int8_model)
    print("Save int8 model to {}".format(int8_model))

if __name__ == "__main__":
    main()
