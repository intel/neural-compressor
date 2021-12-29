import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as torch_models
import torch.nn.functional as F
import torch.optim as optim


class ConvNetMaker(nn.Module):
	"""
	Creates a simple (plane) convolutional neural network
	"""
	def __init__(self, layers):
		"""
		Makes a cnn using the provided list of layers specification
		The details of this list is available in the paper
		:param layers: a list of strings, representing layers like ["CB32", "CB32", "FC10"]
		"""
		super(ConvNetMaker, self).__init__()
		self.conv_layers = []
		self.fc_layers = []
		h, w, d = 32, 32, 3
		previous_layer_filter_count = 3
		previous_layer_size = h * w * d
		num_fc_layers_remained = len([1 for l in layers if l.startswith('FC')])
		for layer in layers:
			if layer.startswith('Conv'):
				filter_count = int(layer[4:])
				self.conv_layers += [nn.Conv2d(previous_layer_filter_count, filter_count, kernel_size=3, padding=1),
									 nn.BatchNorm2d(filter_count), nn.ReLU(inplace=True)]
				previous_layer_filter_count = filter_count
				d = filter_count
				previous_layer_size = h * w * d
			elif layer.startswith('MaxPool'):
				self.conv_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
				h, w = int(h / 2.0), int(w / 2.0)
				previous_layer_size = h * w * d
			elif layer.startswith('FC'):
				num_fc_layers_remained -= 1
				current_layer_size = int(layer[2:])
				if num_fc_layers_remained == 0:
					self.fc_layers += [nn.Linear(previous_layer_size, current_layer_size)]
				else:
					self.fc_layers += [nn.Linear(previous_layer_size, current_layer_size), nn.ReLU(inplace=True)]
				previous_layer_size = current_layer_size
		
		conv_layers = self.conv_layers
		fc_layers = self.fc_layers
		self.conv_layers = nn.Sequential(*conv_layers)
		self.fc_layers = nn.Sequential(*fc_layers)
	
	def forward(self, x):
		x = self.conv_layers(x)
		x = x.view(x.size(0), -1)
		x = self.fc_layers(x)
		return x



plane_cifar10_book = {
	'2': ['Conv16', 'MaxPool', 'Conv16', 'MaxPool', 'FC10'],
	'4': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'FC10'],
	'6': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC10'],
	'8': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool',
		  'Conv128', 'Conv128','MaxPool', 'FC64', 'FC10'],
	'10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
		   'Conv256', 'Conv256', 'Conv256', 'Conv256' , 'MaxPool', 'FC128' ,'FC10'],
}


plane_cifar100_book = {
	'2': ['Conv32', 'MaxPool', 'Conv32', 'MaxPool', 'FC100'],
	'4': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC100'],
	'6': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool','Conv128', 'Conv128' ,'FC100'],
	'8': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
		  'Conv256', 'Conv256','MaxPool', 'FC64', 'FC100'],
	'10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
		   'Conv256', 'Conv256', 'Conv256', 'Conv256' , 'MaxPool', 'FC512', 'FC100'],
}