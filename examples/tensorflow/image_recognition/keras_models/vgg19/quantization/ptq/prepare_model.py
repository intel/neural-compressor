import argparse
from tensorflow.keras.applications.vgg19 import VGG19
def get_vgg19_model(saved_path):
    model = VGG19(weights='imagenet')
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
    get_vgg19_model(args.output_model)
