import numpy as np
import time
import argparse
import os
import json
import torch

import alexnet


def infer_perf(index, model_file):
    train_loader, test_loader = alexnet.data_loader(batch_size=10000)

    if index=='8':
        model = alexnet.load_int8_mod(model_file)
    else:
        model = alexnet.load_mod(model_file)

    accuracy = 0
    test_loss, accuracy = alexnet.do_test_mod(model, test_loader)
    print('accuracy:', accuracy)

         
    throughput = 0
    times = 10
    warmup = 2
    infer_time = 0.0
    with torch.no_grad():
      model.eval()
      for i in range(times):
          bt = time.time()
          for images, labels in test_loader:
            log_ps = model(images)
          et = time.time()
          if i>=warmup:
            infer_time += (et-bt)
            
      print("batch_size {}".format(test_loader.batch_size))
      throughput = test_loader.batch_size* len(test_loader) / (et - bt) /(times-warmup)
      print('max throughput(fps):', throughput)

    # latency when BS=1
    warmup = len(test_loader)*0.2
    bt = 0
    infer_time = 0.0
    train_loader, test_loader = alexnet.data_loader(batch_size=1)
    
    for i,(images, labels) in enumerate(test_loader):        
        bt = time.time()
        log_ps = model(images)
        et = time.time()
        if i >= warmup:
            infer_time += (et-bt)
        
    latency = infer_time * 1000 / (len(test_loader) - warmup)
    print("run times {}".format(times-warmup))
    print('latency(ms):', latency)

    return accuracy, throughput, latency


def save_res(index, result):
    accuracy, throughput, latency = result
    res = {}
    res['accuracy'] = accuracy
    res['throughput'] = throughput
    res['latency'] = latency

    outfile = index + ".json"
    with open(outfile, 'w') as f:
        json.dump(res, f)
        print("Save result to {}".format(outfile))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, help='file name of output', required=True)

    parser.add_argument('--input-graph', type=str, help='file name for graph', required=True)

    args = parser.parse_args()
    
    save_res(args.index, infer_perf(args.index, args.input_graph))

if __name__ == "__main__":
    main()
