#!/usr/bin/python
# encoding: utf-8
import os
from torch.utils.data import Dataset
from PIL import Image


class GTResDataset(Dataset):

	def __init__(self, root_path, gt_dir=None, transform=None, transform_train=None):
		self.pairs = []
		for f in zip(os.listdir(root_path),os.listdir(gt_dir)):
			image_path = os.path.join(root_path, f[0])
			gt_path = os.path.join(gt_dir, f[1])
			if f[0].endswith(".jpg") or f[0].endswith(".png") or f[1].endswith(".png") or f[1].endswith(".jpg"):
				self.pairs.append([image_path, gt_path.replace('.png', '.jpg'), None])
		self.transform = transform
		self.transform_train = transform_train

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, index):
		from_path, to_path, _ = self.pairs[index]
		from_im = Image.open(from_path).convert('RGB')
		to_im = Image.open(to_path).convert('RGB')

		if self.transform:
			to_im = self.transform(to_im)
			from_im = self.transform(from_im)

		return from_im, to_im
