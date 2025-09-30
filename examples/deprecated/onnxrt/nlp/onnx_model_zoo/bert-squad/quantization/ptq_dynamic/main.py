import numpy as np
import onnxruntime
import onnx
import tokenization
import os
from run_onnx_squad import *
import json
from run_onnx_squad import read_squad_examples, convert_examples_to_features, write_predictions
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
from squad_evaluate import evaluate
import sys

max_seq_length = 256
doc_stride = 128
max_query_length = 64
n_best_size = 20
max_answer_length = 30

class squadDataset(Dataset):
    def __init__(self, unique_ids, input_ids, input_mask, segment_ids, bs):
        self.unique_ids = unique_ids
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.bs = bs

    def __getitem__(self, index):
        return (list(range(index, index + self.bs)), self.input_ids[index:index + self.bs][0].astype(np.int64), 
            self.input_mask[index:index + self.bs][0].astype(np.int64), self.segment_ids[index:index + self.bs][0].astype(np.int64)), 0

    def __len__(self):
        assert len(self.input_ids) == len(self.input_mask)
        assert len(self.input_ids) == len(self.segment_ids)
        return len(self.input_ids)

def evaluate_squad(model, dataloader, input_ids, eval_examples, extra_data, input_file):
    session = onnxruntime.InferenceSession(model.SerializeToString(), None,
        providers=onnxruntime.get_available_providers())
    for output_meta in session.get_outputs():
        print(output_meta)
    for input_meta in session.get_inputs():
        print(input_meta)
    n = len(input_ids)
    bs = 1
    all_results = []
    start = timer()
    for idx, (batch, label) in tqdm.tqdm(enumerate(dataloader), desc="eval"):
        data = {"unique_ids_raw_output___9:0": np.array(batch[0], dtype=np.int64),
                "input_ids:0": np.array(batch[1], dtype=np.int64),
                "input_mask:0": np.array(batch[2], dtype=np.int64),
                "segment_ids:0": np.array(batch[3], dtype=np.int64)}
        result = session.run(["unique_ids:0","unstack:0", "unstack:1"], data)
        in_batch = result[0].shape[0]
        start_logits = [float(x) for x in result[1][0].flat]
        end_logits = [float(x) for x in result[2][0].flat]
        for i in range(0, in_batch):
            unique_id = len(all_results)
            all_results.append(RawResult(unique_id=unique_id, start_logits=start_logits,end_logits=end_logits))
    
    # postprocessing
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    output_prediction_file = os.path.join(output_dir, "predictions_mobilebert_fp32.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions_mobilebert_fp32.json")
    write_predictions(eval_examples, extra_data, all_results,
                    n_best_size, max_answer_length,
                    True, output_prediction_file, output_nbest_file)

    with open(input_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        expected_version = '1.1'
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                    ', but got dataset with v-' + dataset_json['version'],
                    file=sys.stderr)
        dataset = dataset_json['data']
    with open(output_prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    res = evaluate(dataset, predictions)
    return res['exact_match']

def main():
    parser = argparse.ArgumentParser(description='onnx squad')
    parser.add_argument('--model_path', required=True, help='model path')
    parser.add_argument('--save_path', type=str, default='bertsquad_tune.onnx', 
                        help='save tuned model path')
    parser.add_argument('--data_path', type=str,
                        help='datseset path')
    parser.add_argument('--tune', action='store_true', default=False, 
                        help='run neural_compressor tune')
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='run benchmark')
    parser.add_argument('--mode', type=str, default='performance',
                        help="benchmark mode of performance or accuracy")
    parser.add_argument('--benchmark_nums', type=int, default=1000,
                        help="Benchmark numbers of samples")
    parser.add_argument('--batch_size', type=int, default=1, 
                        help="batch size for benchmark")
    args = parser.parse_args()

    model = onnx.load(args.model_path)

    predict_file = 'dev-v1.1.json'
    input_file=os.path.join(args.data_path, predict_file)
    eval_examples = read_squad_examples(input_file=input_file)

    vocab_file = os.path.join('uncased_L-12_H-768_A-12', 'vocab.txt')
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(eval_examples, tokenizer, 
                                                                            max_seq_length, doc_stride, max_query_length)

    dataset = squadDataset(eval_examples, input_ids, input_mask, segment_ids, 1) 
    eval_dataloader = DataLoader(dataset, batch_size=args.batch_size)

    def eval_func(model):
        return evaluate_squad(model, eval_dataloader, input_ids, eval_examples, extra_data, input_file)

    if args.tune:
        from neural_compressor import quantization, PostTrainingQuantConfig
        config = PostTrainingQuantConfig(approach='dynamic')
        q_model = quantization.fit(model, 
                                   config,
                                   eval_func=eval_func)
        q_model.save(args.save_path)

    if args.benchmark:
        model = onnx.load(args.model_path)
        if args.mode == 'performance':
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            conf = BenchmarkConfig(iteration=100,
                                cores_per_instance=4,
                                num_of_instance=1)
            fit(model, conf, b_dataloader=eval_dataloader)
        elif args.mode == 'accuracy':
            acc_result = eval_func(model)
            print("Batch size = %d" % args.batch_size)
            print("Accuracy: %.5f" % acc_result)


if __name__ == "__main__":
    main()
