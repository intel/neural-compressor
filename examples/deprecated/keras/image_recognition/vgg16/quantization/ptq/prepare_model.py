import argparse
from tensorflow.keras.applications.vgg16 import VGG16
def get_vgg16_model(saved_path):
    model = VGG16(weights='imagenet')
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
    get_vgg16_model(args.output_model)
