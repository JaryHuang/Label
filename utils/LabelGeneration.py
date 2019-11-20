
import random
from .utils import writetxt
from tqdm import tqdm
import torch 
from config import opt
import os


class LabelGeneration(object):
    def __init__(self, class_num, negative_num = 1):
        self.NLNum = negative_num
        self.ClassNum = class_num
     
    def GetNL(self, label):
        return [self.__GenerateLabel(l) for l in label]
        
    def __GenerateLabel(self, l):
        sellabel = [i for i in range(self.ClassNum) if i != l]
        return [random.choice(sellabel) for i in range(self.NLNum)]



    
def label_updata(model,dataloader,root, ori_dataset, res_dataset):
    for step,batch  in enumerate(tqdm(dataloader,desc='label update', unit='batch')):
        result_list = []
        with torch.no_grad():
            data,true_label,noisy_label,name  =  batch
            if opt.use_gpu:
                data =  data.cuda()
            one_hot = torch.Tensor([[0 if i!=j else 1 for i in range(opt.num_classes)] for j in noisy_label]).cuda()
            score= model(data)
            score = torch.nn.functional.softmax(score,1)
            actually_score = score[one_hot.byte()]
            _ , preds = torch.max(score , 1)
            
            for i in range(len(true_label)):
                if actually_score[i]<0.5:
                    result_list.append([name[i],true_label[i],preds[i].item()])
                else:
                    result_list.append([name[i],true_label[i],noisy_label])
    writetxt(result_list,root+res_dataset)
