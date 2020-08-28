import os
import argparse
import enum
import tarfile
import abc

def get_pretrained_model(destination):
    """
    Obtains a ready to use style_transfer model file.
    Args:
        destination: path to where the file should be stored
    """
    url = "https://storage.googleapis.com/download.magenta.tensorflow.org/models/ \
           arbitrary_style_transfer.tar.gz"

    os.system("curl -o arbitrary_style_transfer.tar.gz {0}".format(url))
    with tarfile.open("arbitrary_style_transfer.tar.gz") as tar:
        if not os.path.exists(destination):
            os.makedirs(destination)
        tar.extractall(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare pre-trained model for style transfer model')
    parser.add_argument('--model_path', type=str, default='./model', help='directory to put models, default is ./model')

    args = parser.parse_args()
    model_path = args.model_path
    try:
        get_pretrained_model(model_path)
    except AttributeError:
        print("The model fetched failed.")

