import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-predefined-models',
  templateUrl: './predefined-models.component.html',
  styleUrls: ['./predefined-models.component.scss', './../start-page/start-page.component.scss']
})
export class PredefinedModelsComponent implements OnInit {

  modelList = {
    "tensorflow": {
      "resnet50v1.0": {
        "model_src_dir": "image_recognition",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget --continue https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb", "resnet50_fp32_pretrained_model.pb", "/tmp/pre-trained-models/resnet50/fp32/freezed_resnet50.pb"],
        "yaml": "resnet50_v1.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "resnet50v1.5": {
        "model_src_dir": "image_recognition",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget https://zenodo.org/record/2535873/files/resnet50_v1.pb", "resnet50_v1.pb", "/tmp/pre-trained-models/resnet50v1_5/fp32/resnet50_v1.pb"],
        "yaml": "resnet50_v1_5.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "resnet101": {
        "model_src_dir": "image_recognition",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet101_fp32_pretrained_model.pb", "resnet101_fp32_pretrained_model.pb", "/tmp/pre-trained-models/resnet101/fp32/optimized_graph.pb"],
        "yaml": "resnet101.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "resnet_v1_50_slim": {
        "model_src_dir": "image_recognition/slim",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz", "resnet_v1_50.ckpt", "/tmp/tensorflow/slim/resnet_v1_50.ckpt"],
        "yaml": "resnet_v1_50.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "resnet_v1_101_slim": {
        "model_src_dir": "image_recognition/slim",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz", "resnet_v1_101.ckpt", "/tmp/tensorflow/slim/resnet_v1_101.ckpt"],
        "yaml": "../resnet101.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "resnet_v1_152_slim": {
        "model_src_dir": "image_recognition/slim",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz", "resnet_v1_152.ckpt", "/tmp/tensorflow/slim/resnet_v1_152.ckpt"],
        "yaml": "resnet_v1_152.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "inception_v1_slim": {
        "model_src_dir": "image_recognition/slim",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz", "inception_v1.ckpt", "/tmp/tensorflow/slim/inception_v1.ckpt"],
        "yaml": "../inceptionv1.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "inception_v2_slim": {
        "model_src_dir": "image_recognition/slim",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz", "inception_v2.ckpt", "/tmp/tensorflow/slim/inception_v2.ckpt"],
        "yaml": "../inceptionv2.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "inception_v3": {
        "model_src_dir": "image_recognition",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/inceptionv3_fp32_pretrained_model.pb", "inceptionv3_fp32_pretrained_model.pb", "/tmp/pre-trained-models/inceptionv3/fp32/freezed_inceptionv3.pb"],
        "yaml": "inceptionv3.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "inception_v3_slim": {
        "model_src_dir": "image_recognition/slim",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz", "inception_v3.ckpt", "/tmp/tensorflow/slim/inception_v3.ckpt"],
        "yaml": "inceptionv3.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "inception_v4": {
        "model_src_dir": "image_recognition",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/inceptionv4_fp32_pretrained_model.pb", "inceptionv4_fp32_pretrained_model.pb", "/tmp/pre-train-model-slim/pbfile/frozen_pb/frozen_inception_v4.pb"],
        "yaml": "inceptionv4.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "inception_v4_slim": {
        "model_src_dir": "image_recognition/slim",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz", "inception_v4.ckpt", "/tmp/tensorflow/slim/inception_v4.ckpt"],
        "yaml": "../inceptionv4.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "vgg16_slim": {
        "model_src_dir": "image_recognition/slim",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz", "vgg_16.ckpt", "/tmp/tensorflow/slim/vgg_16.ckpt"],
        "yaml": "../vgg16.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "vgg19_slim": {
        "model_src_dir": "image_recognition/slim",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz", "vgg_19.ckpt", "/tmp/tensorflow/slim/vgg_19.ckpt"],
        "yaml": "../vgg19.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "resnetv2_50_slim": {
        "model_src_dir": "image_recognition/slim",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz", "resnet_v2_50.ckpt", "/tmp/tensorflow/slim/resnet_v2_50.ckpt"],
        "yaml": "../resnet_v2_50.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "resnetv2_101_slim": {
        "model_src_dir": "image_recognition/slim",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz", "resnet_v2_101.ckpt", "/tmp/tensorflow/slim/resnet_v2_101.ckpt"],
        "yaml": "../resnet_v2_101.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "resnetv2_152_slim": {
        "model_src_dir": "image_recognition/slim",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": ["wget http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz", "resnet_v2_152.ckpt", "/tmp/tensorflow/slim/resnet_v2_152.ckpt"],
        "yaml": "../resnet_v2_152.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "mobilenetv1": {
        "model_src_dir": "image_recognition",
        "dataset_location": "/tmp/dataset/imagenet",
        "input_model": [
          "wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb",
          "mobilenet_v1_1.0_224_frozen.pb",
          "/tmp/pre-train-model-slim/pbfile/frozen_pb/frozen_mobilenet_v1.pb"
        ],
        "yaml": "mobilenet_v1.yaml",
        "strategy": "basic",
        "batch_size": 100
      },

      "ssd_resnet50_v1": {
        "model_src_dir": "object_detection",
        "dataset_location": "/tf_dataset/tensorflow/coco_val.record",
        "input_model": [
          "wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz",
          "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb",
          "/tmp/pre-train-model-oob/object_detection/ssd_resnet50_v1/frozen_inference_graph.pb"
        ],
        "yaml": "ssd_resnet50_v1.yaml",
        "strategy": "basic",
        "batch_size": 1
      },
      "ssd_mobilenet_v1": {
        "model_src_dir": "object_detection",
        "dataset_location": "/tf_dataset/tensorflow/coco_val.record",
        "input_model": [
          "wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz",
          "ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb",
          "/tmp/pre-train-model-oob/object_detection/ssd_mobilenet_v1/frozen_inference_graph.pb"
        ],
        "yaml": "ssd_mobilenet_v1.yaml",
        "strategy": "basic",
        "batch_size": 1
      },
      "faster_rcnn_inception_resnet_v2": {
        "model_src_dir": "object_detection",
        "dataset_location": "/tf_dataset/tensorflow/coco_val.record",
        "input_model": [
          "wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz",
          "faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb",
          "/tmp/pre-train-model-oob/object_detection/faster_rcnn_inception_resnet_v2/frozen_inference_graph.pb"
        ],
        "yaml": "faster_rcnn_inception_resnet_v2.yaml",
        "strategy": "basic",
        "batch_size": 1
      },
      "faster_rcnn_resnet101": {
        "model_src_dir": "object_detection",
        "dataset_location": "/tf_dataset/tensorflow/coco_val.record",
        "input_model": [
          "wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz",
          "faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb",
          "/tmp/pre-train-model-oob/object_detection/faster_rcnn_resnet101/frozen_inference_graph.pb"
        ],
        "yaml": "faster_rcnn_resnet101.yaml",
        "strategy": "basic",
        "batch_size": 1
      },
      "mask_rcnn_inception_v2": {
        "model_src_dir": "object_detection",
        "dataset_location": "/tf_dataset/tensorflow/coco_val.record",
        "input_model": [
          "wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz",
          "mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb",
          "/tmp/pre-train-model-oob/object_detection/mask_rcnn_inception_v2/frozen_inference_graph.pb"
        ],
        "yaml": "mask_rcnn_inception_v2.yaml",
        "strategy": "basic",
        "batch_size": 1
      },
      "wide_deep_large_ds": {
        "model_src_dir": "recommendation/wide_deep_large_ds",
        "dataset_location": "/tf_dataset/tensorflow/wide_deep_large_ds/dataset",
        "input_model": [
          "wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/wide_deep_fp32_pretrained_model.pb",
          "wide_deep_fp32_pretrained_model.pb",
          "/tmp/tensorflow/wide_deep_large_ds/fp32_optimized_graph.pb"
        ],
        "yaml": "wide_deep_large_ds.yaml",
        "strategy": "basic",
        "batch_size": 256
      },
      "style_transfer": {
        "model_src_dir": "style_transfer",
        "dataset_location": "style_images,content_images",
        "input_model": [
          "wget https://storage.googleapis.com/download.magenta.tensorflow.org/models/arbitrary_style_transfer.tar.gz",
          "arbitrary_style_transfer/model.ckpt*",
          "/tmp/tensorflow/style_transfer/arbitrary_style_transfer/model.ckpt"
        ],
        "yaml": "conf.yaml",
        "strategy": "basic",
        "batch_size": -1
      },
      "bert": {
        "model_src_dir": "nlp/bert",
        "dataset_location": "/tf_dataset/tensorflow/bert/data",
        "input_model": [
          "wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip",
          "bert_large_checkpoints/model.ckpt*",
          "/tmp/tensorflow/bert/model"
        ],
        "yaml": "bert.yaml",
        "strategy": "basic",
        "batch_size": 1
      },
      "deeplabv3": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/semantic_segmentation/deeplab/v3/deeplabv3.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "efficientnet-b0": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/classification/efficientnet/b0/tf/efficientnet-b0.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "efficientnet-b0_auto_aug": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/classification/efficientnet/b0_auto_aug/tf/efficientnet-b0_auto_aug.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "efficientnet-b5": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/classification/efficientnet/b5/tf/efficientnet-b5.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "faster_rcnn_inception_v2_coco": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/object_detection/common/faster_rcnn/faster_rcnn_inception_v2_coco/tf/faster_rcnn_inception_v2_coco.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "faster_rcnn_resnet101_ava_v2.1": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/faster_rcnn_resnet101_ava_v2.1_2018_04_30/frozen_inference_graph.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "faster_rcnn_resnet101_coco": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/object_detection/common/faster_rcnn/faster_rcnn_resnet101_coco/tf/faster_rcnn_resnet101_coco.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "faster_rcnn_resnet101_kitti": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/faster_rcnn_resnet101_kitti_2018_01_28/frozen_inference_graph.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "faster_rcnn_resnet101_lowproposals_coco": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/frozen_inference_graph.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "faster_rcnn_resnet50_coco": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/object_detection/common/faster_rcnn/faster_rcnn_resnet50_coco/tf/faster_rcnn_resnet50_coco.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "faster_rcnn_resnet50_lowproposals_coco": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/object_detection/common/faster_rcnn/faster_rcnn_resnet50_lowproposals_coco/tf/faster_rcnn_resnet50_lowproposals_coco.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "googlenet-v1": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/classification/googlenet/v1/tf/googlenet-v1.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "googlenet-v3": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/classification/googlenet/v3/tf/googlenet-v3.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "googlenet-v4": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/classification/googlenet/v4/tf/googlenet-v4.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "inception-resnet-v2": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/classification/inception-resnet/v2/tf/inception-resnet-v2.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 100
      },
      "resnet-50": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/classification/resnet/v1/50/tf/official/resnet-50.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 1
      },
      "rfcn-resnet101-coco": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/object_detection/common/rfcn/rfcn_resnet101_coco/tf/rfcn-resnet101-coco.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 1
      },
      "vehicle-license-plate-detection-barrier-0123": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/object_detection/barrier/tf/0123/vehicle-license-plate-detection-barrier-0123.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 1
      },
      "yolo-v2": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/object_detection/yolo/yolo_v2/tf/yolo-v2.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 1
      },
      "yolo-v3": {
        "model_src_dir": "oob_models",
        "dataset_location": "",
        "input_model": "/tmp/tensorflow/tf_oob_models/object_detection/yolo/yolo_v3/tf/yolo-v3.pb",
        "yaml": "config.yaml",
        "strategy": "basic",
        "batch_size": 1
      }
    }
  };

  constructor() { }

  ngOnInit() {
  }

  objectKeys(obj): string[] {
    return Object.keys(obj);
  }

}
