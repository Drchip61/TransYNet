import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
from my_scheduler import LR_Scheduler
from swin_ynet import Encoder
from data.dataset_GZ_224  import MyData
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

import pytorch_ssim
import pytorch_iou
warnings.filterwarnings("ignore")
model = Encoder().cuda()

deal = nn.Sigmoid()
#model.load_state_dict(torch.load('ynet_swinB_4444_pp_levir_final.pth'))

'''
path = 'ynet_swinB_4444_edge.pth'

save_model = torch.load(path)
model_dict =  model.state_dict()
state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
#print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
model_dict.update(state_dict)
model.load_state_dict(model_dict)
'''
#model.load_state_dict(torch.load('model256_2.pth'))

#model.load_from(weights=np.load(CONFIGS['R50-ViT-B_16'].pretrained_path))

model = model.train()
#bce_loss = nn.BCELoss(size_average = True)
ce_loss = nn.CrossEntropyLoss()
#ssim_loss = pytorch_ssim.SSIM(window_size = 14,size_average = True)
#iou_loss = pytorch_iou.IOU(size_average = True)

LR = 0.1
LR_VGG=0.00001
EPOCH = 100
#optimizer= optim.AdamW(model.parameters(), lr=LR,weight_decay=0.01,betas=(0.9,0.999))
scheduler = LR_Scheduler('cos',LR,EPOCH,3743//5+1)

optimizer= optim.SGD(model.parameters(), lr=LR,momentum=0.9,weight_decay=0.0005,nesterov=False)
def make_optimizer(LR,model):
    params = []
    for key,value in model.named_parameters():
        #print(key)
        if not value.requires_grad:
            continue
        if "encoder1" in key:
            #print(key)
            lr = LR * 0.1
        else:
            #print(key)
            lr = LR
        params += [{"params":[value],"lr":lr}]
    optimizer = getattr(torch.optim,"SGD")(params,momentum=0.9,weight_decay=0.0005,nesterov=False)
    return optimizer 



train_loader= DataLoader(MyData(),
                      shuffle=True,
                      batch_size=1,
                      pin_memory=True,
                      num_workers=8,
                      )


losses0 = 0
losses1 = 0
losses2 = 0
losses3 = 0


print(len(train_loader))
def adjust_learning_rate(optimizer,epoch,start_lr):
    if epoch%20 == 0:  #epoch != 0 and 
    #lr = start_lr*(1-epoch/EPOCH)
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"]*0.1
        print(param_group["lr"])
for epoch_num in range(EPOCH):
    print(epoch_num)
    adjust_learning_rate(optimizer,epoch_num,LR)
    print('LR is:',optimizer.state_dict()['param_groups'][0]['lr'])
    show_dict = {'epoch':epoch_num}
    for i_batch,(im1,im2,label0,label1,label2,label3) in enumerate(tqdm.tqdm(train_loader,ncols=60,postfix=show_dict)):  #,edge0,edge1,edge2,edge3
        im1 = im1.cuda()
        im2 = im2.cuda()
        label0 = label0.cuda()
        label1 = label1.cuda()
        label2 = label2.cuda()
        label3 = label3.cuda()
        
        outputs = model(im1,im2)
        
        loss0 = ce_loss(outputs[0],label0.long())
        loss1 = ce_loss(outputs[1],label1.long())
        loss2 = ce_loss(outputs[2],label2.long())
        loss3 = ce_loss(outputs[3],label3.long())
       
        loss = loss0+loss1+loss2+loss3
       
        
        losses0 += loss0
        losses1 += loss1
        losses2 += loss2
        losses3 += loss3
        
        
        optimizer.zero_grad()
        #scheduler(optimizer,i_batch,epoch_num)
        loss.backward()
        optimizer.step()
        if i_batch%100 == 0:
            print(i_batch,'|','losses0: {:.3f}'.format(losses0.data),'|','losses1: {:.3f}'.format(losses1.data),'|','losses2: {:.3f}'.format(losses2.data),'|','losses3: {:.3f}'.format(losses3.data))

            
            losses0=0
            losses1=0
            losses2=0
            losses3=0
            
            
            
       
    torch.save(model.state_dict(),'ynet_swinB_pp_GZ_224.pth')
