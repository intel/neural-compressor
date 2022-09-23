import argparse
from tensorflow.keras.applications.inception_v3 import InceptionV3
def get_inception_v3_model(saved_path):
    model = InceptionV3(weights='imagenet')
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
    get_inception_v3_model(args.output_model)
