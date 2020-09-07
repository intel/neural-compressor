from utils import *

BATCH_SIZE = 1
PATH_TO_MODEL = "./"

models = [

    # aipg-vdcnn
    {
        'model_name': 'aipg-vdcnn',
        'input': {'input_x': generate_data([1024, 70]), 'phase': False},
        'output': ['fc-3/predictions']
    },

    # arttrack-coco-multi
    {
        'model_name': 'arttrack-coco-multi',
        'input': {'sub': generate_data([688,688,3])},
        'output': ['pose/part_pred/block4/BiasAdd','pose/locref_pred/block4/BiasAdd','pose/pairwise_pred/block4/BiasAdd']
    },

    # arttrack-mpii-single
    {
        'model_name': 'arttrack-mpii-single',
        'input': {'sub': generate_data([688, 688, 3])},
        'output': ['pose/part_pred/block4/BiasAdd','pose/locref_pred/block4/BiasAdd']
    },

    # ava-person-vehicle-detection-stage1-2.0.0
    {
        'model_name': 'ava-person-vehicle-detection-stage1-2.0.0',
        'input': {'data': generate_data([544, 992, 3]), 'im_info': generate_data([4])},
        'output': ['cls_score','bbox_pred','head_score','head_bbox_pred','torso_score','torso_reg_pred','wp_score','wp_reg_pred']
    },

    # cpm-pose
    {
        'model_name': 'cpm-pose',
        'input': {'CPM/Placeholder_1': generate_data([368, 368, 3]), 'CPM/Placeholder_2': generate_data([368, 368, 1])},
        'output': ['CPM/PoseNet/Mconv7_stage6/Conv2D']
    },

    # deepspeech
    {
        'model_name': 'deepspeech',
        'input': {'input_node': generate_data([16, 19, 26]), 'previous_state_h/read': generate_data([2048]), 'previous_state_c/read': generate_data([2048])},
        'output': ['raw_logits','lstm_fused_cell/GatherNd','lstm_fused_cell/GatherNd_1']
    },

    # deepvariant_wgs
    {
        'model_name': 'deepvariant_wgs',
        'input': {'InceptionV3/InceptionV3/Conv2d_1a_3x3/ReadVariableOp': generate_data([100, 221, 6])},
        'output': ['InceptionV3/Predictions/Reshape_1']
    },

    # dense_vnet_abdominal_ct
    {
        'model_name': 'dense_vnet_abdominal_ct',
        'input': {'worker_0/validation/Squeeze': generate_data([144, 144, 144, 1])},
        'output': ['worker_0/DenseVNet/trilinear_resize_3/transpose_1']
    },

    # east_resnet_v1_50
    {
        'model_name': 'east_resnet_v1_50',
        'input': {'input_images': generate_data([1024, 1920, 3])},
        'output': ['feature_fusion/Conv_7/Sigmoid','feature_fusion/concat_3']
    },

    # facenet-20180408-102900
    {
        'model_name': 'facenet-20180408-102900',
        'input': {'image_batch': generate_data([160, 160, 3]), 'phase_train': False},
        'output': ['embeddings']
    },

    # faster-rcnn-resnet101-coco-sparse-60-0001
    {
        'model_name': 'faster-rcnn-resnet101-coco-sparse-60-0001',
        'input': {'image': generate_data([800, 1280, 3])},
        'output': ['openvino_outputs/cls_score','openvino_outputs/bbox_pred']
    },

    # handwritten-score-recognition-0003
    {
        'model_name': 'handwritten-score-recognition-0003',
        'input': {'Placeholder': generate_data([32, 64, 1])},
        'output': ['shadow/LSTMLayers/transpose_time_major']
    },

    # license-plate-recognition-barrier-0007
    {
        'model_name': 'license-plate-recognition-barrier-0007',
        'input': {'input': generate_data([24, 94, 3])},
        'output': ['d_predictions']
    },

    # ncf
    {
        'model_name': 'ncf',
        'input': {'0': False, '1': False},
        'output': ['add_2']
    },

    # optical_character_recognition-text_recognition-tf
    {
        'model_name': 'optical_character_recognition-text_recognition-tf',
        'input': {'input': generate_data([32, 100, 3])},
        'output': ['shadow/LSTMLayers/transpose_time_major']
    },

    # pose-ae-multiperson
    {
        'model_name': 'pose-ae-multiperson',
        'input': {'my_model/strided_slice': generate_data([512, 512, 3])},
        'output': ['my_model/out_3/add']
    },

    # pose-ae-refinement
    {
        'model_name': 'pose-ae-refinement',
        'input': {'my_model/strided_slice': generate_data([512, 512, 3])},
        'output': ['my_model/out_3/add']
    },

    # PRNet
    {
        'model_name': 'PRNet',
        'input': {'Placeholder': generate_data([256, 256, 3])},
        'output': ['resfcn256/Conv2d_transpose_16/Sigmoid']
    },

    # text-recognition-0012
    {
        'model_name': 'text-recognition-0012',
        'input': {'Placeholder': generate_data([32, 120, 1])},
        'output': ['shadow/LSTMLayers/transpose_time_major']
    },

    # vggvox
    {
        'model_name': 'vggvox',
        'input': {'input': generate_data([512, 1000, 1])},
        'output': ['fc8/BiasAdd']
    },

    # efficientnet
    {
        'model_name': 'efficientnet-b0',
        'input': {'sub': generate_data([224, 224, 3])},
        'output': ['logits']
    },

    {
        'model_name': 'efficientnet-b0_auto_aug',
        'input': {'sub': generate_data([224, 224, 3])},
        'output': ['logits']
    },

    {
        'model_name': 'efficientnet-b5',
        'input': {'sub': generate_data([224, 224, 3])},
        'output': ['logits']
    },

    {
        'model_name': 'efficientnet-b7_auto_aug',
        'input': {'sub': generate_data([224, 224, 3])},
        'output': ['logits']
    },

    # yolo_v3_mlp
    {
        'model_name': 'yolo_v3_mlp',
        'input': {'input/input_data': generate_data([416, 416, 3])},
        'output': ['pred_sbbox/concat_2', 'pred_mbbox/concat_2', 'pred_lbbox/concat_2']
    },

    # Hierarchical_LSTM
    {
        'model_name': 'Hierarchical_LSTM',
        'input': {'batch_in': generate_data([1024, 27]), 'batch_out': generate_data([1024, 27])},
        'output': ['map_1/while/output_module_vars/prediction']
    },

    # Resnetv2_200
    {
        'model_name': 'Resnetv2_200',
        'input': {'fifo_queue_Dequeue': generate_data([224, 224, 3])},
        'output': ['resnet_v2_200/SpatialSqueeze']
    },
    {
        'model_name': 'resnet50',
        'input': {'map/TensorArrayStack/TensorArrayGatherV3': generate_data([224, 224, 3])},
        'output': ['softmax_tensor']
    },
    # TextCNN
    {
        'model_name': 'TextCNN',
        'input': {'is_training_flag': False, 'input_x': generate_data([200,], input_dtype="int32"), 'dropout_keep_prob': generate_data([384])},
        'output': ['Sigmoid']
    },
    # TextRNN
    {
        'model_name': 'TextRNN',
        'input': {'input_x': generate_data([100,], input_dtype="int32"), 'input_y': generate_data([100,], input_dtype="int32"), 'dropout_keep_prob': generate_data([100])},
        'output': ['Accuracy']
    },
    # TextRCNN, need bs=512
    {
        'model_name': 'TextRCNN',
        'input': {'input_x': generate_data([100,], input_dtype="int32"), 'input_y': generate_data([1,], input_dtype="int32"), 'dropout_keep_prob': generate_data([300])},
        'output': ['Accuracy']
    },
    # efficientnet-b0
    {
        'model_name': 'efficientnet-b0',
        'input': {'sub': generate_data([224, 224, 3])},
        'output': ['logits']
    },
    # efficientnet-b0
    {
        'model_name': 'efficientdet-b0',
        'input': {'input': generate_data([512, 512, 3])},
        'output': ['class_net/class-predict/BiasAdd', 'class_net/class-predict_1/BiasAdd', 'class_net/class-predict_2/BiasAdd', 'class_net/class-predict_3/BiasAdd', 'class_net/class-predict_4/BiasAdd', 'box_net/box-predict/BiasAdd', 'box_net/box-predict_1/BiasAdd', 'box_net/box-predict_2/BiasAdd', 'box_net/box-predict_3/BiasAdd', 'box_net/box-predict_4/BiasAdd']
    },
    # vggvox
    {
        'model_name': 'vggvox',
        'input': {'sub': generate_data([512, 1000, 1])},
        'output': ['fc8/BiasAdd']
    },

]
