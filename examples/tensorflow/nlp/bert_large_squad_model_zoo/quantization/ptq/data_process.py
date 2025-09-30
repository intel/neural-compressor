#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json
import re
import sys
import string
import collections

import numpy as np
import tensorflow as tf

from abc import abstractmethod
from collections import Counter
from neural_compressor.tensorflow.utils.data import default_collate, BaseDataLoader, BatchSampler, IterableFetcher

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Calculate the max metric for each ground truth.

    For each answer in ground_truths, evaluate the metric of prediction with
    this answer, and return the max metric.

    Args:
        metric_fn: The function to calculate the metric.
        prediction: The prediction result.
        ground_truths: A list of correct answers.

    Returns:
        The max metric. Float point number.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def normalize_answer(text: str) -> str:
    """Normalize the answer text.

    Lower text, remove punctuation, articles and extra whitespace,
    and replace other whitespace (newline, tab, etc.) to space.

    Args:
        s: The text to be normalized.

    Returns:
        The normalized text.
    """

    def _remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def _white_space_fix(text):
        return " ".join(text.split())

    def _remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def _lower(text):
        return text.lower()

    return _white_space_fix(_remove_articles(_remove_punc(_lower(text))))

def exact_match_score(prediction, ground_truth):
    """Compute the exact match score between prediction and ground truth.

    Args:
        prediction: The result of predictions to be evaluated.
        ground_truth: The ground truth.

    Returns:
        The exact match score.
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    """Calculate the F1 score of the prediction and the ground_truth.

    Args:
        prediction: The predicted result.
        ground_truth: The ground truth.

    Returns:
        The F1 score of prediction. Float point number.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate(dataset, predictions):
    """Evaluate the average F1 score and the exact match score for Question-Answering results.

    Args:
        dataset: The dataset to evaluate the prediction. A list instance of articles.
          An article contains a list of paragraphs, a paragraph contains a list of
          question-and-answers (qas), and a question-and-answer contains an id, a question,
          and a list of correct answers. For example:
        predictions: The result of predictions to be evaluated. A dict mapping the id of
          a question to the predicted answer of the question.

    Returns:
        The F1 score and the exact match score.
    """
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = "Unanswered question " + qa["id"] + " will receive score 0."
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}


class BaseMetric(object):
    """The base class of Metric."""

    def __init__(self, metric, single_output=False, hvd=None):
        """Initialize the basic metric.

        Args:
            metric: The metric class.
            single_output: Whether the output is single or not, defaults to False.
            hvd: The Horovod class for distributed training, defaults to None.
        """
        self._metric_cls = metric
        self._single_output = single_output
        self._hvd = hvd

    def __call__(self, *args, **kwargs):
        """Evaluate the model predictions, and the reference.

        Returns:
            The class itself.
        """
        self._metric = self._metric_cls(*args, **kwargs)
        return self

    @abstractmethod
    def update(self, preds, labels=None, sample_weight=None):
        """Update the state that need to be evaluated.

        Args:
            preds: The prediction result.
            labels: The reference. Defaults to None.
            sample_weight: The sampling weight. Defaults to None.

        Raises:
            NotImplementedError: The method should be implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Clear the predictions and labels.

        Raises:
            NotImplementedError: The method should be implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def result(self):
        """Evaluate the difference between predictions and labels.

        Raises:
            NotImplementedError: The method should be implemented by subclass.
        """
        raise NotImplementedError

    @property
    def metric(self):
        """Return its metric class.

        Returns:
            The metric class.
        """
        return self._metric

    @property
    def hvd(self):
        """Return its hvd class.

        Returns:
            The hvd class.
        """
        return self._hvd

    @hvd.setter
    def hvd(self, hvd):
        """Set its hvd.

        Args:
            hvd: The Horovod class for distributed training.
        """
        self._hvd = hvd


class SquadF1(BaseMetric):
    """Evaluate for v1.1 of the SQuAD dataset."""

    def __init__(self):
        """Initialize the score list."""
        self._score_list = []  # squad metric only work when all data preds collected

    def update(self, preds, labels, sample_weight=None):
        """Add the predictions and labels.

        Args:
            preds: The predictions.
            labels: The labels corresponding to the predictions.
            sample_weight: The sample weight.
        """
        if preds:
            if getattr(self, "_hvd", None) is not None:
                gathered_preds_list = self._hvd.allgather_object(preds)
                gathered_labels_list = self._hvd.allgather_object(labels)
                temp_preds_list, temp_labels_list = [], []
                for i in range(0, self._hvd.size()):
                    temp_preds_list += gathered_preds_list[i]
                    temp_labels_list += gathered_labels_list[i]
                preds = temp_preds_list
                labels = temp_labels_list
            result = evaluate(labels, preds)
            self._score_list.append(result["f1"])

    def reset(self):
        """Reset the score list."""
        self._score_list = []

    def result(self):
        """Compute F1 score."""
        if len(self._score_list) == 0:
            return 0.0
        return np.array(self._score_list).mean()


class ParseDecodeBert:
    """Helper function for TensorflowModelZooBertDataset.

    Parse the features from sample.
    """

    def __call__(self, sample):
        """Parse the sample data.

        Args:
            sample: Data to be parsed.
        """
        # Dense features in Example proto.
        feature_map = {
            "input_ids": tf.compat.v1.VarLenFeature(dtype=tf.int64),
            "input_mask": tf.compat.v1.VarLenFeature(dtype=tf.int64),
            "segment_ids": tf.compat.v1.VarLenFeature(dtype=tf.int64),
        }

        features = tf.io.parse_single_example(sample, feature_map)

        input_ids = features["input_ids"].values
        input_mask = features["input_mask"].values
        segment_ids = features["segment_ids"].values

        return (input_ids, input_mask, segment_ids)


class TFDataLoader(object):  # pragma: no cover
    """Tensorflow dataloader class.

    In tensorflow1.x dataloader is coupled with the graph, but it also support feed_dict
    method to do session run, this dataloader is designed to satisfy the usage of feed dict
    in tf1.x. Although it's a general dataloader and can be used in MXNet and PyTorch.

    Args:
        dataset: obj. wrapper of needed data.
        batch_size: int. batch size
    """

    def __init__(self, dataset, batch_size=1, last_batch="rollover"):
        """Initialize `TFDataDataLoader` class."""
        self.dataset = dataset
        self.last_batch = last_batch
        self.batch_size = batch_size
        dataset = dataset.batch(batch_size)

    def batch(self, batch_size, last_batch="rollover"):
        """Dataset return data per batch."""
        drop_last = False if last_batch == "rollover" else True
        self.batch_size = batch_size
        self.dataset = self.dataset.batch(batch_size, drop_last)

    def __iter__(self):
        """Iterate dataloader."""
        return self._generate_dataloader(
            self.dataset,
            batch_size=self.batch_size,
            last_batch=self.last_batch,
        )

    def _generate_dataloader(
        self,
        dataset,
        batch_size=1,
        last_batch="rollover",
        collate_fn=None,
        sampler=None,
        batch_sampler=None,
        num_workers=None,
        pin_memory=None,
        distributed=False,
    ):
        """Yield data."""
        drop_last = False if last_batch == "rollover" else True

        def check_dynamic_shape(element_spec):
            if isinstance(element_spec, collections.abc.Sequence):
                return any([check_dynamic_shape(ele) for ele in element_spec])
            elif isinstance(element_spec, tf.TensorSpec):
                return True if element_spec.shape.num_elements() is None else False
            else:
                raise ValueError("unrecognized element spec...")

        def squeeze_output(output):
            if isinstance(output, collections.abc.Sequence):
                return [squeeze_output(ele) for ele in output]
            elif isinstance(output, np.ndarray):
                return np.squeeze(output, axis=0)
            else:
                raise ValueError("not supported output format....")

        if tf.executing_eagerly():
            index = 0
            outputs = []
            for iter_tensors in dataset:
                samples = []
                iter_inputs, iter_labels = iter_tensors[0], iter_tensors[1]
                if isinstance(iter_inputs, tf.Tensor):
                    samples.append(iter_inputs.numpy())
                else:
                    samples.append(tuple(iter_input.numpy() for iter_input in iter_inputs))
                if isinstance(iter_labels, tf.Tensor):
                    samples.append(iter_labels.numpy())
                else:
                    samples.append([np.array(l) for l in iter_labels])
                index += 1
                outputs.append(samples)
                if index == batch_size:
                    outputs = default_collate(outputs)
                    yield outputs
                    outputs = []
                    index = 0
            if len(outputs) > 0:
                outputs = default_collate(outputs)
                yield outputs
        else:
            try_single_batch = check_dynamic_shape(dataset.element_spec)
            dataset = dataset.batch(1 if try_single_batch else batch_size, drop_last)
            ds_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            iter_tensors = ds_iterator.get_next()
            data_config = tf.compat.v1.ConfigProto()
            data_config.use_per_session_threads = 1
            data_config.intra_op_parallelism_threads = 1
            data_config.inter_op_parallelism_threads = 16
            data_sess = tf.compat.v1.Session(config=data_config)
            # pylint: disable=no-name-in-module
            from tensorflow.python.framework.errors_impl import OutOfRangeError

            while True:
                if not try_single_batch:
                    try:
                        outputs = data_sess.run(iter_tensors)
                        yield outputs
                    except OutOfRangeError:
                        data_sess.close()
                        return
                else:
                    try:
                        outputs = []
                        for i in range(0, batch_size):
                            outputs.append(squeeze_output(data_sess.run(iter_tensors)))
                        outputs = default_collate(outputs)
                        yield outputs
                    except OutOfRangeError:
                        if len(outputs) == 0:
                            data_sess.close()
                            return
                        else:
                            outputs = default_collate(outputs)
                            yield outputs
                            data_sess.close()
                            return


class ModelZooBertDataset(object):
    """Tensorflow dataset for three-input Bert in tf record format.

    Root is a full path to tfrecord file, which contains the file name.
    Please use Resize transform when batch_size > 1
    Args: root (str): path of dataset.
          label_file (str): path of label file.
          task (str, default='squad'): task type of model.
          model_type (str, default='bert'): model type, support 'bert'.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according.
    """

    def __init__(self, root, label_file, task="squad", model_type="bert", transform=None, filter=None, num_cores=28):
        """Initialize the attributes of class."""
        with open(label_file) as lf:
            label_json = json.load(lf)
            assert label_json["version"] == "1.1", "only support squad 1.1"
            self.label = label_json["data"]

        record_iterator = tf.compat.v1.python_io.tf_record_iterator(root)
        example = tf.train.SequenceExample()
        for element in record_iterator:
            example.ParseFromString(element)
            break
        feature = example.context.feature
        if len(feature["input_ids"].int64_list.value) == 0 and len(feature["input_mask"].int64_list.value) == 0:
            raise ValueError(
                "Tfrecord format is incorrect, please refer\
                'https://github.com/tensorflow/models/blob/master/research/\
                object_detection/dataset_tools/' to create correct tfrecord"
            )
        # pylint: disable=no-name-in-module
        from tensorflow.python.data.experimental import parallel_interleave

        tfrecord_paths = [root]
        ds = tf.data.TFRecordDataset.list_files(tfrecord_paths)
        ds = ds.apply(
            parallel_interleave(
                tf.data.TFRecordDataset,
                cycle_length=num_cores,
                block_length=5,
                sloppy=True,
                buffer_output_elements=10000,
                prefetch_input_elements=10000,
            )
        )
        if transform is not None:
            transform.transform_list.insert(0, ParseDecodeBert())
        else:
            transform = ParseDecodeBert()
        ds = ds.map(transform, num_parallel_calls=None)
        if filter is not None:
            ds = ds.filter(filter)
        ds = ds.prefetch(buffer_size=1000)
        ds = TFDataLoader(ds)
        self.root = []
        for inputs in ds:
            self.root.append(inputs)
        self.transform = transform
        self.filter = filter

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        return self.root[index], self.label

    def __len__(self):
        """Length of the dataset."""
        return len(self.root)


class TFSquadV1PostTransform(object):
    """Postprocess the predictions of bert on SQuAD.

    Args:
        label_file (str): path of label file
        vocab_file(str): path of vocabulary file
        n_best_size (int, default=20):
            The total number of n-best predictions to generate in nbest_predictions.json
        max_seq_length (int, default=384):
            The maximum total input sequence length after WordPiece tokenization.
            Sequences longer than this will be truncated, shorter than this will be padded
        max_query_length (int, default=64):
            The maximum number of tokens for the question.
            Questions longer than this will be truncated to this length
        max_answer_length (int, default=30):
            The maximum length of an answer that can be generated. This is needed because
            the start and end predictions are not conditioned on one another
        do_lower_case (bool, default=True):
            Whether to lower case the input text.
            Should be True for uncased models and False for cased models
        doc_stride (int, default=128):
            When splitting up a long document into chunks,
            how much stride to take between chunks

    Returns:
        tuple of processed prediction and label
    """

    def __init__(
        self,
        label_file,
        vocab_file,
        n_best_size=20,
        max_seq_length=384,
        max_query_length=64,
        max_answer_length=30,
        do_lower_case=True,
        doc_stride=128,
    ):
        """Initialize `TFSquadV1PostTransform` class."""
        from tokenization import FullTokenizer
        from create_tf_record import read_squad_examples, convert_examples_to_features
        self.eval_examples = read_squad_examples(label_file)
        tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        self.eval_features = []

        def append_feature(feature):
            self.eval_features.append(feature)

        convert_examples_to_features(
            examples=self.eval_examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            output_fn=append_feature,
        )

        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.do_lower_case = do_lower_case
        self.RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

    def process_result(self, results):
        """Get the processed results."""
        processed_results = []
        # notice the result list sequence
        for unique_id, start_logits, end_logits in zip(*results):
            processed_results.append(
                self.RawResult(
                    unique_id=int(unique_id),
                    start_logits=[float(x) for x in start_logits.flat],
                    end_logits=[float(x) for x in end_logits.flat],
                )
            )

        return processed_results

    def get_postprocess_result(self, sample):
        """Get the post processed results."""
        if sample == (None, None):
            return (None, None)
        all_results, label = sample
        all_results = self.process_result(all_results)
        example_index_to_features = collections.defaultdict(list)
        for feature in self.eval_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
        )

        all_predictions = collections.OrderedDict()
        for example_index, example in enumerate(self.eval_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min mull score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            for feature_index, feature in enumerate(features):
                # skip the case that is not predicted
                if feature.unique_id not in unique_id_to_result:
                    all_predictions[example.qas_id] = "*#skip this example#*"
                    continue
                result = unique_id_to_result[feature.unique_id]
                start_indexes = TFSquadV1PostTransform._get_best_indexes(result.start_logits, self.n_best_size)
                end_indexes = TFSquadV1PostTransform._get_best_indexes(result.end_logits, self.n_best_size)

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > self.max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index],
                            )
                        )

                prelim_predictions = sorted(
                    prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True
                )
                _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "NbestPrediction", ["text", "start_logit", "end_logit"]
                )

                seen_predictions = {}
                nbest = []
                for pred in prelim_predictions:
                    if len(nbest) >= self.n_best_size:
                        break
                    feature = features[pred.feature_index]
                    if pred.start_index > 0:  # this is a non-null prediction
                        tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                        orig_doc_start = feature.token_to_orig_map[pred.start_index]
                        orig_doc_end = feature.token_to_orig_map[pred.end_index]
                        orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
                        tok_text = " ".join(tok_tokens)

                        # De-tokenize WordPieces that have been split off.
                        tok_text = tok_text.replace(" ##", "")
                        tok_text = tok_text.replace("##", "")

                        # Clean whitespace
                        tok_text = tok_text.strip()
                        tok_text = " ".join(tok_text.split())
                        orig_text = " ".join(orig_tokens)

                        final_text = TFSquadV1PostTransform.get_final_text(tok_text, orig_text, self.do_lower_case)
                        if final_text in seen_predictions:
                            continue

                        seen_predictions[final_text] = True
                    else:
                        final_text = ""
                        seen_predictions[final_text] = True

                    nbest.append(
                        _NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit)
                    )

                # In very rare edge cases we could have no valid predictions. So we
                # just create a nonce prediction in this case to avoid failure.
                if not nbest:
                    nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

                assert len(nbest) >= 1

                total_scores = []
                best_non_null_entry = None
                for entry in nbest:
                    total_scores.append(entry.start_logit + entry.end_logit)
                    if not best_non_null_entry:
                        if entry.text:
                            best_non_null_entry = entry
                probs = TFSquadV1PostTransform._compute_softmax(total_scores)

                nbest_json = []
                for i, entry in enumerate(nbest):
                    output = collections.OrderedDict()
                    output["text"] = entry.text
                    output["probability"] = probs[i]
                    output["start_logit"] = entry.start_logit
                    output["end_logit"] = entry.end_logit
                    nbest_json.append(output)

                assert len(nbest_json) >= 1
                all_predictions[example.qas_id] = nbest_json[0]["text"]
        return (all_predictions, label)

    @staticmethod
    def _get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    @staticmethod
    def _compute_softmax(scores):
        """Compute softmax probability over raw logits."""
        import math

        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs

    @staticmethod
    def get_final_text(pred_text, orig_text, do_lower_case):
        """Project the tokenized prediction back to the original text."""
        import six

        from tokenization import BasicTokenizer

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for i, c in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        tok_text = " ".join(tokenizer.tokenize(orig_text))
        start_position = tok_text.find(pred_text)
        if start_position == -1:
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            return orig_text

        tok_s_to_ns_map = {}
        for i, tok_index in six.iteritems(tok_ns_to_s_map):
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            return orig_text

        output_text = orig_text[orig_start_position : (orig_end_position + 1)]
        return output_text

    def __call__(self, sample):
        """Call the get_postprocess_result."""
        return self.get_postprocess_result(sample)


class CollectTransform(object):
    """Postprocess the predictions, collect data."""

    def __init__(self, length=10833):
        """Initialize `CollectTransform` class."""
        self.length = length
        self.unique_id = []
        self.start_logits = []
        self.end_logits = []
        self.all_sample = (None, None)
        self.idx = 1000000000

    def __call__(self, sample):
        """Collect postprocess data."""
        all_results, label = sample
        result_list = [np.expand_dims(result, 0) for result in all_results]
        for result in result_list:
            if len(self.unique_id) < self.length:
                result = result.transpose(2, 0, 1)
                self.unique_id.append(self.idx)
                self.start_logits.append(result[0])
                self.end_logits.append(result[1])
                self.idx += 1
        if len(self.unique_id) == self.length:
            self.all_sample = ([self.unique_id, self.start_logits, self.end_logits], label)
        return self.all_sample


class TFModelZooCollectTransform(CollectTransform):
    """Postprocess the predictions of model zoo, collect data."""

    def __call__(self, sample):
        """Collect postprocess data."""
        all_results, label = sample
        if len(all_results) == 1:
            all_results = all_results.reshape((2, 1, 384))
        all_results = zip(all_results[0], all_results[1])
        for start_logits, end_logits in all_results:
            if len(self.unique_id) < self.length:
                self.unique_id.append(self.idx)
                self.start_logits.append(start_logits)
                self.end_logits.append(end_logits)
                self.idx += 1
        if len(self.unique_id) == self.length:
            self.all_sample = ([self.unique_id, self.start_logits, self.end_logits], label)
        return self.all_sample


class TFSquadV1ModelZooPostTransform(TFSquadV1PostTransform):
    """Postprocess the predictions of bert on SQuADV1.1.

    See class TFSquadV1PostTransform for more details
    """

    def __init__(
        self,
        label_file,
        vocab_file,
        n_best_size=20,
        max_seq_length=384,
        max_query_length=64,
        max_answer_length=30,
        do_lower_case=True,
        doc_stride=128,
    ):
        """Initialize `TFSquadV1ModelZooPostTransform` class."""
        super().__init__(
            label_file,
            vocab_file,
            n_best_size,
            max_seq_length,
            max_query_length,
            max_answer_length,
            do_lower_case,
            doc_stride,
        )
        self.length = len(self.eval_features)
        self.collect_data = TFModelZooCollectTransform(length=self.length)

    def __call__(self, sample):
        """Collect data and get postprocess results."""
        sample = self.collect_data(sample)
        return self.get_postprocess_result(sample)


class ModelZooBertDataLoader(BaseDataLoader):  # pragma: no cover
    """This dataloader is designed to satisfy the usage of Model Zoo Bert models."""

    def _generate_dataloader(
        self,
        dataset,
        batch_size,
        last_batch,
        collate_fn,
        sampler,
        batch_sampler,
        num_workers,
        pin_memory,
        shuffle,
        distributed,
    ):
        def bert_collate_fn(batch):
            input_ids = []
            input_mask = []
            segment_ids = []
            for elem in batch:
                input_ids.append(elem[0][0][0])
                input_mask.append(elem[0][1][0])
                segment_ids.append(elem[0][2][0])
            inputs = [input_ids, input_mask, segment_ids]
            return inputs, batch[0][1]

        drop_last = False if last_batch == "rollover" else True
        sampler = self._generate_sampler(dataset, distributed)
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.fetcher = IterableFetcher(dataset, bert_collate_fn, drop_last, distributed)

        inputs = []
        for batched_indices in self.batch_sampler:
            try:
                data = self.fetcher(batched_indices)
                yield data
            except StopIteration:
                return
