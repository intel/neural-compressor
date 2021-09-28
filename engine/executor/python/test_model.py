import argparse
import numpy as np
from engine_py import Model

parser = argparse.ArgumentParser(description='Deep engine Model Executor')
parser.add_argument('--weight',  default='', type=str, help='weight of the model')
parser.add_argument('--config',  default='', type=str, help='config of the model')
args = parser.parse_args()

input_0 = np.random.randint(0,384,(64,171)).reshape(64,171)
input_1 = np.random.randint(0,2,(64, 171)).reshape(64,171)
input_2 = np.random.randint(0,2,(64, 171)).reshape(64,171)
softmax_min = np.array([0])
softmax_max = np.array([1])

model = Model(args.config, args.weight)
# for fp32
out = model.forward([input_0, input_1, input_2])
# for int8
# output = model.forward([input_0, input_1, input_2, softmax_min, softmax_max])
output = output[0].reshape(64,171,2)

print('input value is')
print(input_0)
print(input_1)
print(input_2)

print('output value is')
print(output)
