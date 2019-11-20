# coding: utf8
import torch
import warnings

class DefaultConfig(object):
   #'ResNet18','ResNet50',"ResNext"
   model_name = "ResNet18"
   
   batch_size =128 
   num_workers = 8
   pic_size = 32 
   # Cifar,ImageNet
   dataset="Cifar" 
   train_root = './data/cifar10/'
   train_name="cifar10_noisy.txt"
   
   val_root = './data/'
   val_name = dataset+"/val.txt" 

   test_root = './data/cifar10/'
   test_name="cifar10_noisy.txt" 

   num_classes = 10
   num_nagetive = 1
   NL_EPOCH = 100
   SelNL_EPOCH = 100
   SelPL_EPOCH = 100

   start_epoch = 0
   max_epoch = NL_EPOCH + SelNL_EPOCH + SelPL_EPOCH
   
   #'cross_entropy','SelPLNL'
   label_mode = 'SelNLPL'

   #lr = [0.1, 0.02, 0.01,0.005] 
   lr = [0.1, 0.01, 0.001] 
   #lr_decay = 0.1
   weight_decay = 1e-4
   #milestones = [60,120,160]

   extra_inform = "test"
   env = '{}_{}_{}_epoch_{}_{}'.format(model_name,dataset,max_epoch,label_mode,extra_inform) 

   #load_model_path = "checkpoints/ResNet18_Cifar_300_epoch_SelPLNL_Negetive=10/299.ckpt"
   load_model_path = None 

   use_gpu = torch.cuda.is_available() 
   
   save_checkpoint = 'checkpoints/'+env
  
   result_name = "result/SelNLPL_negative=10.csv"
   def parse(self, kwargs):
      for k, v in kwargs.items():
         if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" %k)
         setattr(self, k, v)
         
      print('user config:')
      for k, v in self.__class__.__dict__.items():
         if not k.startswith('__'):
            print(k, getattr(self, k))

opt = DefaultConfig()
