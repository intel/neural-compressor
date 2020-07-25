import os
import argparse
import mxnet as mx
from gluoncv import utils
from gluoncv.model_zoo import get_model

def convert_from_gluon(model_name, image_shape, model_path, classes=1000):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, model_path)

    print('Converting model from Gluon-CV ModelZoo %s... into path %s' % (model_name, model_path))
    net = get_model(name=model_name, classes=classes, pretrained=True)
    net.hybridize()
    x = mx.sym.var('data')
    y = net(x)
    y = mx.sym.SoftmaxOutput(data=y, name='softmax')
    symnet = mx.symbol.load_json(y.tojson()) # pylint: disable=no-member
    params = net.collect_params()
    args = {}
    auxs = {}
    for param in params.values():
        v = param._reduce()
        k = param.name
        if 'running' in k:
            auxs[k] = v
        else:
            args[k] = v
    mod = mx.mod.Module(symbol=symnet, context=mx.cpu(),
                        label_names = ['softmax_label'])
    mod.bind(for_training=False,
             data_shapes=[('data', (1,) +
                          tuple([int(i) for i in image_shape.split(',')]))])
    mod.set_params(arg_params=args, aux_params=auxs)
    dst_dir = os.path.join(dir_path, 'model')
    prefix = os.path.join(dir_path, 'model', model_name)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    mod.save_checkpoint(prefix, 0)
    print("Save model {} at {}".format(model_name, model_path))
    return prefix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare pre-trained model for MXNet ImageNet Classifier')
    parser.add_argument('--model_name', type=str, default='resnet18_v1', help='model to download, default is resnet18_v1',
                        choices=["resnet18_v1", "resnet50_v1", "squeezenet1.0", "mobilenet1.0", "mobilenetv2_1.0", "inceptionv3"])
    parser.add_argument('--model_path', type=str, default='./model', help='directory to put models, default is ./model')

    args = parser.parse_args()
    model_name = args.model_name
    model_path = args.model_path
    image_shape = '3,299,299' if model_name == 'inceptionv3' else '3,224,224'
    
    convert_from_gluon(model_name, image_shape, model_path)

