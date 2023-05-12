import cv2
import onnx
import os
import numpy as np
from sklearn.model_selection import KFold
import sklearn
import pickle
import logging
import argparse
import onnxruntime as ort

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

def load_bin(path, image_size):
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data_list = []
    for flip in [0,1]:
        data = np.empty((len(issame_list)*2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = cv2.imdecode(np.frombuffer(_bin, np.uint8), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, axes=(2, 0, 1))
        for flip in [0,1]:
            if flip==1:
                img = np.flip(img, axis=2)
            data_list[flip][i][:] = img
    data = np.concatenate(data_list)
    return (data.astype('float32'), issame_list)

def load_property(data_dir):
    for line in open(os.path.join(os.path.split(data_dir)[0], 'property')):
        vec = line.strip().split(',')
        assert len(vec)==3
        image_size = [int(vec[1]), int(vec[2])]
    return image_size
    
class LFold:
    def __init__(self, n_splits = 2, shuffle = False):
        self.n_splits = n_splits
        if self.n_splits>1:
            self.k_fold = KFold(n_splits = n_splits, shuffle = shuffle)

    def split(self, indices):
        if self.n_splits>1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)
    
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
    return accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
  
    acc = float(tp+tn)/dist.size
    return acc

class Metric:
    def __init__(self, nfolds):
        self.embeddings = []
        self.actual_issame = None
        self.nfolds = nfolds

    def update(self, preds, labels):
        if self.actual_issame is None:
            self.actual_issame = np.asarray(labels)
            self.actual_issame = self.actual_issame.squeeze()
        self.embeddings.append(preds[0])

    def reset(self):
        self.embeddings = []
        self.actual_issame = None

    def result(self):
        embeddings_list = np.array(self.embeddings)
        num = embeddings_list.shape[0]
        embeddings = embeddings_list[:num//2] + embeddings_list[num//2:]
        embeddings = embeddings.squeeze()
        embeddings = sklearn.preprocessing.normalize(embeddings)
        thresholds = np.arange(0, 4, 0.02)
        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]
        accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
            np.asarray(self.actual_issame), nrof_folds=self.nfolds)
        return np.mean(accuracy)

class Dataloader:
    def __init__(self, data_dir, batch_size):
        self.batch_size = batch_size
        path = os.path.join(data_dir)
        # Load data
        if os.path.exists(path):
            data_set = load_bin(path, image_size)
            self.data_list = data_set[0]
            self.issame_list = data_set[1]
 
    def __iter__(self):
        for data in self.data_list:
            yield np.expand_dims(data, axis=0), self.issame_list

def eval_func(model, dataloader, metric):
    metric.reset()
    sess = ort.InferenceSession(model.SerializeToString(), providers=ort.get_available_providers())
    ort_inputs = {}
    input_names = [i.name for i in sess.get_inputs()]
    for input_data, label in dataloader:
        output = sess.run(None, dict(zip(input_names, [input_data])))
        metric.update(output, label)
    return metric.result()

if __name__ == '__main__':
    logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
    parser = argparse.ArgumentParser(
        description="ArcFace fine-tune examples for face recognition/verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help="Pre-trained model on onnx file"
    )
    parser.add_argument(
        '--dataset_location',
        type=str,
        help="Imagenet data path"
    )
    parser.add_argument(
        '--benchmark',
        action='store_true', \
        default=False
    )
    parser.add_argument(
        '--tune',
        action='store_true', \
        default=False,
        help="whether quantize the model"
    )
    parser.add_argument(
        '--output_model',
        type=str,
        help="output model path"
    )
    parser.add_argument(
        '--nfolds',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--mode',
        type=str,
        help="benchmark mode of performance or accuracy"
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
    )
    args = parser.parse_args()
    # Load image size
    image_size = load_property(args.dataset_location)
    print('image_size', image_size)
    
    dataloader = Dataloader(args.dataset_location, args.batch_size)
    model = onnx.load(args.model_path)
    metric = Metric(args.nfolds)
    def eval(onnx_model):
        return eval_func(onnx_model, dataloader, metric)
    
    if args.benchmark:
        if args.mode == 'performance':
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            conf = BenchmarkConfig(warmup=10, iteration=1000, cores_per_instance=4, num_of_instance=1)
            fit(model, conf, b_dataloader=dataloader)
        elif args.mode == 'accuracy':
            acc_result = eval(model)
            print("Batch size = %d" % dataloader.batch_size)
            print("Accuracy: %.5f" % acc_result)

    if args.tune:
        from neural_compressor import quantization, PostTrainingQuantConfig
        config = PostTrainingQuantConfig(approach='static')
 
        q_model = quantization.fit(model, config, calib_dataloader=dataloader,
			     eval_func=eval)

        q_model.save(args.output_model)
