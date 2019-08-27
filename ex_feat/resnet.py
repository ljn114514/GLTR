import torch.nn as nn
import math, torch
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from NonLocalBlock1D import NonLocalBlock1D

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=1000, train=True):
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.istrain = train

		self.frames = 16

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
		self.avgpool = nn.AvgPool2d((16,8), stride=1)

		self.num_features = 128
		self.feat = nn.Linear(512 * block.expansion, self.num_features)

		self.feat_bn = nn.BatchNorm1d(self.num_features*4)


		self.feat1 = nn.Conv2d(1, 128, kernel_size=(3,128), stride=1, dilation=(1,1), padding=(1,0), bias=False)
		self.feat2 = nn.Conv2d(1, 128, kernel_size=(3,128), stride=1, dilation=(2,1), padding=(2,0), bias=False)
		self.feat3 = nn.Conv2d(1, 128, kernel_size=(3,128), stride=1, dilation=(3,1), padding=(3,0), bias=False)
		init.normal_(self.feat1.weight, std=0.001)
		init.normal_(self.feat2.weight, std=0.001)
		init.normal_(self.feat3.weight, std=0.001)

		self.Nonlocal_block0 = NonLocalBlock1D(128*4)


	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.feat(x)

		#print x.size()
		x = x.unsqueeze(dim=0)
		#print x.size()
		#x = x.view(x.size(0)/self.frames, self.frames, -1)

		x0 = torch.transpose(x, 1, 2)
		x = x.unsqueeze(dim=1)
		x1 = self.feat1(x).squeeze(dim=3)
		x2 = self.feat2(x).squeeze(dim=3)
		x3 = self.feat3(x).squeeze(dim=3)

		#print x0.size(), x1.size(), x2.size(), x3.size()

		#print x0.size(),x1.size(),x2.size(),x3.size()

		x = torch.cat((x0, x1, x2, x3), dim=1)
		#print x.size()

		x = self.Nonlocal_block0(x).mean(dim=2)


		return x#0+x1+x2+x3

def resnet50(pretrained='True', num_classes=1000, train=True):
	model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, train)
	weight = torch.load(pretrained)
	static = model.state_dict()
	for k in static:
		if k in weight:
			print('load data', k)
			try:
				static[k].copy_(weight[k])
			except :
				print('*'*100)
				print('error %s'%k)
				print (static[k].size(), weight[k].size())
		else:
			print('not in pretrained model')
	return model