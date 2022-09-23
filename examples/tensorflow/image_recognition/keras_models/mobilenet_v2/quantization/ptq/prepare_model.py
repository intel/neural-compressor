import argparse
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
def get_mobilenet_v2_model(saved_path):
    model = MobileNetV2(weights='imagenet')
    model.save(saved_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Export pretained keras model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--output_model',
        type=str,
        help='path to exported model file')

    args = parser.parse_args()
    get_mobilenet_v2_model(args.output_model)
