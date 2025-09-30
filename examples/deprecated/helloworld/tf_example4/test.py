from neural_compressor.data import Datasets
from neural_compressor.data import DataLoader
from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig

def main():
    dataset = Datasets('tensorflow')['dummy_v2']( \
        input_shape=(100, 100, 3), label_shape=(1, ))

    config = PostTrainingQuantConfig(
            inputs=['image_tensor'],
            outputs=['detection_boxes', 'detection_scores', 'detection_classes', 'num_detections'],
            calibration_sampling_size=[20]
            )
    quantized_model = fit(
        model='./model/public/rfcn-resnet101-coco-tf/rfcn_resnet101_coco_2018_01_28/',
        conf=config,
        calib_dataloader=DataLoader(framework='tensorflow', dataset=dataset, batch_size=1))

if __name__ == "__main__":
    main()
