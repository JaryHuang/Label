# coding: utf8
import torch
from PIL import Image
import os

import numpy as np

import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

class DataL(torch.utils.data.Dataset):    
	def __init__(self, root, datatxt, transform=None, mode ='train'):
		self.root = root
		self.transform = transform
		self.mode = mode
		self.dataname = datatxt
		self.img_label = []
		f = open(self.root + datatxt, 'r')
		for line in f:
			words = line.rstrip().split(' ')
			self.img_label.append((words[0],words[1],words[2]))

		#self.img_label = self.img_label[0:300]	

	#第二步装载数据，返回[img,label]
	def __getitem__(self,index):
		
		image_path, true_label, noisy_label =  self.img_label[index][0],self.img_label[index][1],self.img_label[index][2]
		img = Image.open(self.root + image_path).convert('RGB') 

		if self.transform is not None:
			img = self.transform(img)
		if self.mode != "test":
			return img,int(true_label),int(noisy_label)
		else:
			return img,int(true_label),int(noisy_label),image_path

	# return the len of dataset
	def __len__(self):
		return len(self.img_label)

	# describe the dataset information
	def __str__(self):
		print('''The Dataset root is:{}, name:{}, the dataset len is:{}'''
					.format(self.root, self.dataname, len(self.img_label)))
