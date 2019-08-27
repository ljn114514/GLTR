import os, torch, random, cv2, glob
import numpy as np
from torch.utils import data

class videodataset(data.Dataset):
	#mean = np.array([[[104.00699, 116.66877, 122.67892]]])
	def __init__(self, dataset_dir, txt_path, new_height, new_width, frames, transform):
		self.new_height = new_height
		self.new_width = new_width
		self.frames = frames
		self.dataset_dir = dataset_dir
		self.transform = transform

		with open(txt_path) as f:
			line = f.readlines()
			self.img_list = [os.path.join(dataset_dir, i.split()[0])+'*' for i in line]
			self.label_list = [int(i.split()[1]) for i in line]

	def __getitem__(self, index):
		im_dir = self.img_list[index]
		image_list = glob.glob(im_dir)
		image_list.sort()

		images = []
		for name in image_list:
			end = name[-3:]
			if end in ['png', 'jpg', 'jpeg']:
				images.append(name)
		im_paths = images

		frames = []
		for im_path in im_paths:
			image = cv2.imread(im_path)
			image = cv2.resize(image,(self.new_width, self.new_height))
			image = image[:,:, ::-1]
			image = self.transform(image.copy())
			frames.append(image.numpy())

		frames = np.array(frames, np.float32)
		frames = torch.from_numpy(frames).float()
		#print(type(frames), frames.size())

		label = self.label_list[index]
		return frames, label

	def __len__(self):
		return len(self.label_list)