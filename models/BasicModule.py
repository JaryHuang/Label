# coding: utf8
import time
import torch
import torch.nn as nn
import pretrainedmodels
from torchsummary import summary
class BasicModule(torch.nn.Module):
	'''
	封装了nn.Module，主要提供save和load两个方法
	'''
	def __init__(self,opt=None):
		super(BasicModule,self).__init__()
		self.model_name = str(type(self)) # 模型的默认名字

	def load(self, path):
		'''
		可加载指定路径的模型
		'''
		self.load_state_dict(torch.load(path)["net_state_dict"])

	def save(self,save_dir,**kwage):
		torch.save({
				'epoch': epoch,
				'train_loss': train_loss,
				'train_acc': train_acc,
				'test_loss': val_loss,
				'test_acc': val_acc,
				'net_state_dict': net_state_dict},
				save_dir)
		


	

if __name__ == '__main__':
	print(torch.__version__)
	print(pretrainedmodels.model_names)
	model = MyDenseNet()
	#model.cuda()
	summary(model, (3,448, 448))
	#model = MyVgg16Net()
	#print(model)
	#model.load()
