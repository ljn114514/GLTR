import os, random, torch, dataset, cv2, time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import resnet
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

batch_size = 1
##########   DATASET   ###########
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([ transforms.ToTensor(),  normalizer, ])

img_dir = 'video_dataset/Mars/bbox_test/'
test_dataset = dataset.videodataset(dataset_dir=img_dir, txt_path='list_test_seq.txt', new_height=256, new_width=128, frames=16, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
###########   MODEL   ###########

pretrained = '../weight/resnet50_mars_00499.pth'
model = resnet.resnet50(pretrained=pretrained, num_classes=625, train=False)
model.cuda()
model.eval()

name = 'fea'
output = open(name,'w')
num=0
for data in test_loader:		
	num = num+batch_size
	images, label = data

	with torch.no_grad():
		images = Variable(images).cuda()
		images = images.view(images.size(0)*images.size(1), images.size(2), images.size(3), images.size(4))
		#print images.size()	
		out = model(images)
		fea = out.cpu().data
		fea = fea.numpy()

		#print num, np.shape(fea), images.size()

		for j in range(0, np.shape(fea)[0]):
			str1 = ''
			for x in range(len(fea[j])):
				str1 = str1 + str(fea[j][x]) +' '
			str1 = str1 + '\n'
			output.write(str1)
output.close()