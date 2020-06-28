from abc import abstractmethod
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

#from pycocotools import coco
#from pycocotools import cocoeval

'''The metrics supported by iLiT.
   To support new metric, developer just need implement a new subclass in this file.
'''
METRICS = {}

def metric_registry(cls):
    """The class decorator used to register all Metric subclasses.
    
    Returns:
        cls (object): The class of register.
    """
    if cls.__name__.lower() in METRICS:
        raise ValueError('Cannot have two metrics with the same name')
    METRICS[cls.__name__.lower()] = cls
    return cls

class Metric(object):
    """The base class of metrics supported by iLiT.

    Args:
        name (string): The name of supported metric.

    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def compare(self, target):
        """The interface of comparing if metric reaches the goal.

        Args:
            target (Metric): The object of target metric.
        """
        raise notimplementederror

    @abstractmethod
    def evaluate(self, predict, ground_truth):
        """The interface of metric computation.

        Args:
            predict (Tensor): The predict tensor.
            ground_truth (Tensor): The ground truth tensor.

        """
        raise notimplementederror

@metric_registry
class Topk(Metric):
    """The class of calculating topk metric, which usually is used in classification.

    Args:
        topk (dict): The dict of topk for configuration.

    """
    def __init__(self, topk):
        super(Topk, self).__init__('topk')
        assert isinstance(topk, dict)
        assert 'topk' in topk
        assert isinstance(topk['topk'], int)
        self.k = topk['topk']
        self.num_correct = 0
        self.num_sample = 0
        self.acc = 0

    def evaluate(self, predict, label):
        """Compute the Topk metric value of given inputs.

        Args:
            predict (Tensor): The predict tensor.
            ground_truth (Tensor): The ground truth tensor.

        Returns:
            float: topk metric value.

        """

        predict = predict.argsort()[..., -self.k:]
        if self.k == 1:
            correct = accuracy_score(predict, label, normalize=False)
            self.num_correct += correct
            self.num_sample += label.shape[0]

        else:
            for p, l in zip(predict, label):
                # get top-k label with np.argpartition
                # p = np.argpartition(p, -self.k)[-self.k:]
                l = l.astype('int32')
                if l in p:
                    self.num_correct += 1
            self.num_sample += label.shape[0]

        self.acc = self.num_correct / self.num_sample
        return self.acc

    def compare(self, target):
        """Comparing if Topk metric reaches the goal.

        Args:
            target (Metric value): The baseline value of target metric.

        Returns:
            boolen: whether this metric value bigger than target value.

        """
        return self.acc > target.acc


@metric_registry
class CocoMAP(Metric):
    """The class of calculating mAP metric, which usually is used in object detection.

    Args:
        mean_ap(dict): The dict of mean AP for configuration.
    """
    def __init__(self, mean_ap):
        super(MAP, self).__init__('map')
        assert isinstance(mean_ap, dict)
        self.acc = 0
        # List with all ground truths (Ex: [class_ids, bbox])
        self.GroundTruths = []
        # List with all detections (Ex: [class_ids, bbox, confidence])
        self.Detections = []
        # img_id
        self.IMG_ID = 0
        # metrics summary
        self.summary_metrics = {}

    def evaluate(self, detection_info, ground_truth_info):
        """Evaluate mAP between detection and ground_truths.

        Args:
            detection_info (list): list all detection results, each item for each pic, define as:
                        [image_id(optional), detection_classes, detection_boxes, detection_scores]
            ground_truth_info (list): list all ground_truth, each item for each pic, define as:
                        [image_id(optional), groundtruth_classes, groundtruth_boxes]

        Returns:
            float: mAP metric value of the given inputs.
        """
        assert detection_info.shape[0] == ground_truth_info.shape[0], \
            "Pic number of predict should be same as ground_truth!"
        for dt, gd in zip(detection_info, ground_truth_info):

            assert dt[0] == gd[0], 'image id should same between prediction and ground_truth'
            image_id = dt[0]

            self.cocoGtWrapper(image_id=image_id,
                               groundtruth_classes=gd[1],
                               groundtruth_boxes=gd[2])

            self.cocoDtWrapper(image_id=image_id,
                               detection_classes=dt[1],
                               detection_boxes=dt[2],
                               detection_scores=dt[3])

        evaluator = self.COCOEvalWrapper()
        evaluator.evaluate()
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        self.summary_metrics = OrderedDict([('Precision/mAP', evaluator.stats[0]),
                                            ('Precision/mAP@.50IOU', evaluator.stats[1]),
                                            ('Precision/mAP@.75IOU', evaluator.stats[2]),
                                            ('Precision/mAP (small)', evaluator.stats[3]),
                                            ('Precision/mAP (medium)', evaluator.stats[4]),
                                            ('Precision/mAP (large)', evaluator.stats[5]),
                                            ('Recall/AR@1', evaluator.stats[6]),
                                            ('Recall/AR@10', evaluator.stats[7]),
                                            ('Recall/AR@100', evaluator.stats[8]),
                                            ('Recall/AR@100 (small)', evaluator.stats[9]),
                                            ('Recall/AR@100 (medium)', evaluator.stats[10]),
                                            ('Recall/AR@100 (large)', evaluator.stats[11])])

        return evaluator.stats[1]

    def compare(self, target):
        return self.acc > target.acc

    # can use diff mAP metric with diff Wrapper here, eg: VOC, COCO ...
    def COCOEvalWrapper(self,):
        """Wrapper evaluator as a COCOeval.

        Returns:
            evaluator (object): wrappered coco evaluator.
        """
        evaluator = cocoeval.COCOeval(cocoDt=self.cocoDtWrapper, cocoGt=self.cocoGtWrapper)
        return evaluator

    def cocoGtWrapper(self, image_id, groundtruth_classes, groundtruth_boxes):
        """Wrapper inputs as the format of COCOGt

        Args:
            image_id (int): image ids, must align with cocoDt.
            groundtruth_classes (numpy array): ground truth classes of the specific image id.
            groundtruth_boxes (numpy array): ground truth boxes of the specific image id.
        """
        assert len(groundtruth_classes.shape) == 1, \
            'groundtruth_classes is expected to be of rank 1.'
        assert len(groundtruth_boxes.shape) == 2, \
            'groundtruth_boxes is expected to be of rank 2.'
        assert groundtruth_boxes.shape[1] == 4, \
            'groundtruth_boxes should have shape[1] == 4.'

        num_boxes = groundtruth_classes.shape[0]

        for i in range(num_boxes):
            single_gbox_dict = {
                'image_id':     image_id,
                'category_id':  groundtruth_classes[i],
                'bbox':         list(self._ConvertBoxToCOCOFormat(groundtruth_boxes[i, :])),
            }
            self.GroundTruths.append(single_gbox_dict)

    def cocoDtWrapper(self, image_id, detection_classes, detection_boxes, detection_scores):
        """Wrapper inputs as the format of COCODt

        Args:
            image_id (int): image ids, must align with cocoGt.
            detection_classes (numpy array): evaluated classes of the specific image id.
            detection_boxes (numpy array): evaluated boxes of the specific image id.
            detection_scores (numpy array): evaluated scores of each classes.
        """
        assert len(detection_classes.shape) == 1 or len(detection_scores.shape) == 1, \
            'detection_classes is expected to be of rank 1.'
        assert len(detection_boxes.shape) == 2, \
            'detection_boxes is expected to be of rank 2.'
        assert detection_boxes.shape[1] == 4, \
            'detection_boxes should have shape[1] == 4.'

        num_boxes = detection_classes.shape[0]

        for i in range(num_boxes):
            single_dbox_dict = {
                'image_id':     image_id,
                'category_id':  detection_classes[i],
                'bbox':         list(self._ConvertBoxToCOCOFormat(detection_boxes[i, :])),
                'score':        float(detection_scores[i]),
            }
            self.Detections.append(single_dbox_dict)

    def _ConvertBoxToCOCOFormat(self, box):
        """Converts a box in [ymin, xmin, ymax, xmax] format to COCO format.
        This is a utility function for converting from our internal
        [ymin, xmin, ymax, xmax] convention to the convention used by the COCO API
        i.e., [xmin, ymin, width, height].
        Args:
            box (list): a numpy array like [ymin, xmin, ymax, xmax]
        Returns:
            list: list of floats representing [xmin, ymin, width, height]
        """
        return [float(box[1]), float(box[0]), float(box[3] - box[1]),
                float(box[2] - box[0])]

@metric_registry
class F1(Metric):
    """The class of calculating f1 metric, which usually is used in NLP, such as BERT.

    Args:
        f1 (dict): The dict of f1 for configuration.
                    If multi-class, need provide the average config(micro, macro, weighted).
    """
    def __init__(self, f1):
        super(F1, self).__init__('f1')
        assert isinstance(f1, dict)
        self.f1_config = f1
        self.acc = 0

    def evaluate(self, predict, label):
        """Compute the F1 metric value of given inputs

        Args:
            predict (numpy array): predictions.
            label (numpy array): labels.

        Returns:
            acc (float): F1 metric value.
        """
        # binary label
        if predict.shape[1] == 2:
            predict = predict.argmax(axis=-1)
            self.acc = f1_score(predict, label, average='binary')
        # multi label
        else:
            assert 'average' in self.f1_config.keys()
            assert self.f1_config['average'] in ['micro', 'macro', 'weighted']
            predict = predict.argmax(axis=-1)
            self.acc = f1_score(predict, label, average=self.f1_config['average'])
        return self.acc

    def compare(self, target):
        """Comparing if F1 metric reaches the goal.

        Args:
            target (numpy array): target F1 score.

        Returns:
            boolen: If metric acc bigger than target metric acc.
        """
        return self.acc > target.acc


if __name__ == "__main__":
    # tmp test
    print("test acc")
    acc_metric = Topk({'topk':1})
    predict = np.array([[0.22, 0.78], [0.36, 0.64], [0.1, 0.9]])
    label = np.array([1, 0, 1])
    acc = acc_metric.evaluate(predict, label)
    print(acc)

    print("test top-2")
    acc_metric = Topk({'topk':2})
    predict = np.array([[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.2, 0.2, 0.6]])
    label = np.ones([1, 2, 1])
    acc = acc_metric.evaluate(predict, label)
    print(acc)

    print("test f1")
    test_f1 = F1(f1={'average': 'weighted'})
    # binary test
    y_predict = np.array([[0.5, 0.5], [1,0], [0.3,0.7], [0,1], [0.4,0.6], [0.7,0.3]])
    y_true = np.array([0, 0, 1, 0, 0, 1])
    acc = test_f1.evaluate(y_predict, y_true)
    print(acc)
    # multi-label test
    y_predict = np.array([[0.5, 0.3, 0.2], [1, 0, 0], [0.1, 0.3,0.6], [0.2, 0.3, 0.5], [0.1, 0.6,0.3], [0.7,0.1,0.1]])
    y_true = np.array([0, 0, 2, 1, 1, 0])
    acc = test_f1.evaluate(y_predict, y_true)
    print(acc)

    print("test mAP")
    mAP_metric = MAP()
    # [image_id(optional), detection_classes, detection_boxes, detection_scores]
    pd_imgid = 1
    pd_classes = np.array([1,0,0])
    pd_boxes = np.array([[5, 67, 31, 48],
                            [119, 111, 40, 67],
                            [124, 9, 49, 67]])
    pd_scores = np.array([.88, .70, .80])
    predict_1 = np.array([pd_imgid, pd_classes, pd_boxes, pd_scores])

    gd_imgid = 1
    gd_classes = np.array([1,0])
    gd_boxes = np.array([[25, 16, 38, 56],
                         [129, 123, 41, 62]])
    ground_truth_1 = np.array([gd_imgid, gd_classes, gd_boxes])

    predicts = np.array([predict_1])
    ground_truths = np.array([ground_truth_1])

    mAP_5 = mAP_metric.evaluate(predicts, ground_truths)

    print(mAP_5)
