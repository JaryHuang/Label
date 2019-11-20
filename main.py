#coding:utf8
from config import opt

import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from torch.optim import lr_scheduler
import torchvision

from torchnet import meter
from torchsummary import summary
from utils import Visualizer
from tqdm import tqdm
from data import DataL

from loss import CrossEntropyLoss, LabelSmooth, SelNLPL 
from models import Switch_Model

from utils.utils import setup_seed
from data.augment import cifar_transforms, Imagenet_trainsforms
from utils.LabelGeneration import label_updata

setup_seed(19950221)
data_transforms = Imagenet_trainsforms if opt.dataset == "mini-imagenet" else cifar_transforms

def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)
    # step1: model
    model = Switch_Model(opt.model_name, opt.num_classes)
    
    #load model 
    if opt.load_model_path:
        ckpt = torch.load(opt.load_model_path)
        model.load_state_dict(ckpt['net_state_dict'])

    if opt.use_gpu: 
        model.cuda()
        summary(model, (3, opt.pic_size, opt.pic_size))

    # step2: data
    train_data = DataL(root = opt.train_root,datatxt=opt.train_name,
                transform = data_transforms['train'],mode = 'train'
                )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                          shuffle=True, num_workers=opt.num_workers)
   
    if opt.dataset == "Cifar":
        val_data = torchvision.datasets.CIFAR10(root=opt.val_root, train=False,
                                       download=True, transform=data_transforms['val'])
    elif opt.dataset == "cifar100":
        val_data = torchvision.datasets.CIFAR100(root=opt.val_root, train=False,
                                       download=True, transform=data_transforms['val'])
    else:
        raise Exception("Sorry, it is not my dataset {}".format(opt.dataset))

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=opt.batch_size,
                                         shuffle=False, num_workers=opt.num_workers)
    
    dataloaders={'train':train_loader,'val':val_loader}
    dataset_sizes={'train':len(train_data),'val':len(val_data)}

    # step3: loss and optimation
    if opt.label_mode == 'cross_entropy':
        criterion = CrossEntropyLoss(class_num = opt.num_classes,reduction='mean')
        val_criterion = CrossEntropyLoss(class_num = opt.num_classes,reduction='mean')
    elif opt.label_mode == 'SelNLPL':
        criterion = SelNLPL(class_num = opt.num_classes,c = opt.num_classes, r = 0.5, negative_num = opt.num_nagetive)
        val_criterion = CrossEntropyLoss(class_num = opt.num_classes,reduction='mean')
    else:
        raise TypeError("Wrong mode {}, expected: xentropy, soft_bootstrap or hard_bootstrap".format(opt.label_mode))
    

    optimizer = torch.optim.SGD(model.parameters() , 
                            lr =opt.lr[0] , 
                            momentum = 0.9,
                            weight_decay= opt.weight_decay)

    save_dir = opt.save_checkpoint
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    lr = opt.lr[0]
    
    # step4: metric
    confusion_matrix = meter.ConfusionMeter(opt.num_classes)
    best_acc = -0.001
    
    # train
    for epoch in range(opt.start_epoch, opt.max_epoch):
        print('Epoch {}/{}'.format(epoch ,opt.max_epoch - 1))
        print('-' * 10)
        
        if epoch == opt.NL_EPOCH:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr[1] 
                lr = param_group['lr']
        elif epoch == opt.NL_EPOCH + opt.SelNL_EPOCH:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr[2] 
                lr = param_group['lr']
                
        running_loss = 0.0
        running_corrects = 0.0
        model.train(True)
        #_scheduler.step()
    
        for step,(inputs, true_label, label) in enumerate(tqdm(train_loader,desc='Train On {}'.format(opt.dataset), unit='batch')):
            
            labels = label.reshape(len(label),1)
            #print(torch.sum(torch.isnan(inputs)))
            if torch.sum(torch.isnan(inputs)) > 0:
                raise Exception("the input is nan")
            if opt.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

                label = label.cuda()
                true_label = true_label.cuda()
            
            optimizer.zero_grad()   #zero the parameter gradients
            with torch.set_grad_enabled(True):
                outputs= model(inputs)
                if torch.sum(torch.isnan(outputs)) > 0:
                    raise Exception("the output is nan")
                _ , preds = torch.max(outputs , 1)
                if epoch < opt.NL_EPOCH:
                    loss = criterion(outputs , labels, "NL")
                elif epoch < opt.NL_EPOCH + opt.SelNL_EPOCH:
                    loss = criterion(outputs , labels, "SelNL")
                else:
                    loss = criterion(outputs , labels, "SelPL")
         
                loss.backward()  
                
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == label.data)
            
        train_loss = running_loss / dataset_sizes['train']
        train_acc = float(running_corrects) / dataset_sizes['train']
       
        print('Train Loss: {:.4f} top_1_Acc: {:.4f}'.format(train_loss,train_acc))
        model,val_cm,val_loss,val_acc = val(model,val_loader,dataset_sizes['val'],val_criterion)
        print('Val Loss: {:.4f} top_1_Acc: {:.4f}'.format(val_loss,val_acc))
     
        vis.plot_many_stack({'train_loss':train_loss,\
                        'val_loss':val_loss},win_name ="Loss")
        vis.plot_many_stack({'train_acc':train_acc,\
                        'val_acc':val_acc},win_name = 'Acc')

        

        vis.plot_many_stack({'lr':lr},win_name ='lr')
        

        vis.log("epoch:{epoch},\
                train_cm:{train_cm},val_cm:{val_cm}"
       .format(
                epoch = epoch,
                train_cm=str(confusion_matrix.value()),
                val_cm = str(val_cm.value())
                ))
                   
        if val_acc >= best_acc:
            best_acc = val_acc
            best_acc_epoch = epoch
        
        if (epoch+1)%100==0:
            net_state_dict = model.state_dict()
            torch.save({
				'epoch': epoch,
				'train_loss': train_loss,
				'train_acc': train_acc,
				'test_loss': val_loss,
				'test_acc': val_acc,
				'net_state_dict': net_state_dict},
				os.path.join(save_dir, '%03d.ckpt' % epoch))
          
    print('Best val Epoch: {},Best val Acc: {:4f}'.format(best_acc_epoch,best_acc))
    

def val(model,dataloader,data_len,criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    confusion_matrix = meter.ConfusionMeter(opt.num_classes)
    result = []
    for ii, (val_input,val_label) in enumerate(tqdm(dataloader,desc='Val On {}'.format(opt.dataset), unit='batch')):
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
            val_labels = val_label.reshape(len(val_label),1)
        with torch.set_grad_enabled(False):
            score = model(val_input)
            _ , preds = torch.max(score , 1)
            belta_loss= criterion(score, val_labels,None)
            
            confusion_matrix.add(score.data.squeeze(), val_label)
            running_loss += belta_loss.item() * val_input.size(0)
            running_corrects += torch.sum(preds == val_label.data)
    model.train(True)

    cm_value = confusion_matrix.value()
    val_loss = float(running_loss) / data_len
    val_accuracy = float(running_corrects)/data_len
    return model,confusion_matrix, val_loss,val_accuracy

   
def test(**kwargs):

    opt.parse(kwargs)
    defect_count = 0
    model = Switch_Model(opt.model_name, opt.num_classes).eval()
    
    if opt.load_model_path:
        model.load(opt.load_model_path)
    else:
        raise TypeError("Wrong model path of pth")
    if opt.use_gpu: model.cuda()
    
    test_data = DataL(root = opt.test_root,datatxt=opt.test_name,
                transform = data_transforms['val'],mode = 'test')

    test_loader = DataLoader(test_data, batch_size=opt.batch_size,
                                          shuffle=False, num_workers=opt.num_workers)
   
    result_list=[]
    for step,batch in enumerate(tqdm(test_loader,desc='test', unit='batch')):
        with torch.no_grad():
            data,true_label,noisy_label,name  =  batch
            if opt.use_gpu:
                data =  data.cuda()
            one_hot = torch.Tensor([[0 if i!=j else 1 for i in range(opt.num_classes)] for j in noisy_label]).cuda()
            score= model(data)
            score = F.softmax(score,1)
            actually_score = score[one_hot.byte()]
            _ , preds = torch.max(score , 1)
            
            pro = actually_score.to("cpu").numpy()
            preds = preds.to("cpu").numpy()
            for i in range(len(true_label)):
                result_list.append([name[i],true_label[i].item(),preds[i],noisy_label[i].item(),pro[i]])
    data=pd.DataFrame(result_list,columns = ['img_name','true_label','pred_label','noisy_label','pro'])
    data.to_csv(opt.result_name,index=False)



def help():
  
    print('''
    usage : python main.py <function> [--args=value,]
    <function> := train | test | help
    example: 
           python main.py train --env='env0701' --lr=0.01
           python main.py test --dataset='path/to/dataset/root/'
           python main.py help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    import fire
    fire.Fire()

