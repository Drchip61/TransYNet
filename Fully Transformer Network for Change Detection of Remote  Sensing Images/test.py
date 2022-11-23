import torch
import torch.nn as nn
import cv2
#from DUTS_dataset import MyTestData
import numpy as np
import os
import shutil
import sys
from torch.utils.data import DataLoader
#sys.path.append("..")
from data.dataset_WHU_aaai import MyTestData
#from aaai2023 import aaai2023
#from data.dataset import MyTestData
from aaai2023 import aaai2023
#from swin_modify import Encoder_swin
#from swin_modify import Encoder
import torchvision
from utils_conn import *
#model = ViT_seg(CONFIGS['R50-ViT-B_16'],num_classes = 2).cuda()
#model = U_Net().cuda()
model = aaai2023().cuda()
#model = Encoder_swin().cuda()
model.load_state_dict(torch.load('/home/yty/change_detection/gz_1.pth'))
#model.load_state_dict(torch.load('ynet_swinB_4444_GZ.pth'))



test_loader = DataLoader(MyTestData(),shuffle=False,batch_size=1)

outPath = 'cnn_v2_gz'
if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.mkdir(outPath)


with torch.no_grad():
    model = model.eval()
    for i,(im1,im2,label_name) in enumerate(test_loader):
        im1 = im1.cuda()
        im2 = im2.cuda()
        label_name = label_name[0]
        
        outputs = model(im1,im2)
        outputs = outputs[0][0]
        a = bv_test(outputs)
        a = a.ge(0.25001).float()
        #a = outputs[0].unsqueeze(0)

        torchvision.utils.save_image(a,outPath+'/%s'%label_name)

    
        

