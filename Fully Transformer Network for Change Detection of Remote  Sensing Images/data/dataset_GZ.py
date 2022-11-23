import scipy.io as io
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
#from torchvision.transforms import InterpolationMode
import pdb
import os
from PIL import Image
import numpy as np
import os
from utils_conn import sal2conn



class MyData(Dataset):
    def __init__(self):
        super(MyData, self).__init__()
        '''
        self.train_im_path1 = 'data/train_aug256/A_Split'
        self.train_im_path2 = 'data/train_aug256/B_Split'
        self.train_lb_path = 'data/train_aug256/label_Split'
        self.train_lb_path_edge = 'data/train_aug256/edge1'
        self.train_im_num = 8114
        self.train_imgs1 = os.listdir('data/train_aug256/A_Split')
        self.train_imgs2 = os.listdir('data/train_aug256/B_Split')
        self.train_labels = os.listdir('data/train_aug256/label_Split')
        self.train_labels_edge = os.listdir('data/train_aug256/edge1')
        '''
        self.train_im_path1 = 'data/CD_Data_GZ/train/A'
        self.train_im_path2 = 'data/CD_Data_GZ/train/B'
        self.train_lb_path = 'data/CD_Data_GZ/train/label'
        #self.train_lb_path_edge = 'data/train_aug256/edge'
        self.train_im_num = 3743
        self.train_imgs1 = os.listdir(self.train_im_path1)
        self.train_imgs2 = os.listdir(self.train_im_path2 )
        self.train_labels = os.listdir(self.train_lb_path )
        #self.train_labels_edge = os.listdir('data/train_aug256/edge')
        
    def __len__(self):
        return self.train_im_num

    def __getitem__(self, index):
        
        img_file1 = os.path.join(self.train_im_path1,self.train_imgs1[index])
        img1 = Image.open(img_file1)
        img_file2 = os.path.join(self.train_im_path2,self.train_imgs2[index])
        img2 = Image.open(img_file2)
        label_file = os.path.join(self.train_lb_path,self.train_labels[index])
        labels = Image.open(label_file)
        #edge_file = os.path.join(self.train_lb_path_edge,self.train_labels_edge[index])
        #edges = Image.open(edge_file)
        
                 
        im1,im2,lb0,lb1,lb2,lb3 = self.transform(img1,img2,labels)
        lb0 = lb0[0]
        lb1 = lb1[0]
        lb2 = lb2[0]
        lb3 = lb3[0]
        '''
        edge0,edge1,edge2,edge3 = self.transform_edge(edges)
        edge0 = edge0[0]
        edge1 = edge1[0]
        edge2 = edge2[0]
        edge3 = edge3[0]
        '''
  
        return im1,im2, lb0,lb1,lb2,lb3#,edge0,edge1,edge2,edge3

    def transform(self, img1,img2,label):
        
        transform_img = transforms.Compose([#transforms.Resize((14,14),interpolation=InterpolationMode.NEAREST),
                                            transforms.ToTensor(),
                                            ])
        transform_img_4 = transforms.Compose([transforms.Resize((56,56),Image.NEAREST),
                                            transforms.ToTensor(),
                                            ])
        transform_img_8 = transforms.Compose([transforms.Resize((28,28),Image.NEAREST),
                                            transforms.ToTensor(),
                                            ])
        transform_img_16 = transforms.Compose([transforms.Resize((14,14),Image.NEAREST),
                                            transforms.ToTensor(),
                                            ])
        
        transform_img_2 = transforms.Compose([
                                            #transforms.Resize((384,3)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
                                            ])
        im1 = transform_img_2(img1)
        im2 = transform_img_2(img2)
        label0 = transform_img(label)
        label_4 = transform_img_4(label)
        label_8 = transform_img_8(label)
        label_16 = transform_img_16(label)
        return im1,im2,label0,label_4,label_8,label_16
    
    def transform_edge(self, label):
        
        transform_img = transforms.Compose([#transforms.Resize((14,14),interpolation=InterpolationMode.NEAREST),
                                            transforms.ToTensor(),
                                            ])
        transform_img_4 = transforms.Compose([transforms.Resize((56,56),Image.NEAREST),
                                            transforms.ToTensor(),
                                            ])
        transform_img_8 = transforms.Compose([transforms.Resize((28,28),Image.NEAREST),
                                            transforms.ToTensor(),
                                            ])
        transform_img_16 = transforms.Compose([transforms.Resize((14,14),Image.NEAREST),
                                            transforms.ToTensor(),
                                            ])
        
        label0 = transform_img(label)
        label_4 = transform_img_4(label)
        label_8 = transform_img_8(label)
        label_16 = transform_img_16(label)
        return label0,label_4,label_8,label_16
    


class MyTestData(Dataset):
    def __init__(self):
        super(MyTestData, self).__init__()
        self.train_im_path1 = 'data/CD_Data_GZ/GZ_train256/A'
        self.train_im_path2 = 'data/CD_Data_GZ/GZ_train256/B'
        self.train_lb_path = 'data/CD_Data_GZ/GZ_train256/label'
        #self.train_lb_path_edge = 'data/train_aug256/edge'
        self.train_im_num = 2817
        self.train_imgs1 = os.listdir(self.train_im_path1)
        self.train_imgs2 = os.listdir(self.train_im_path2 )
        self.train_labels = os.listdir(self.train_lb_path )

    def __len__(self):
        return self.train_im_num

    def __getitem__(self, index):
        
        img_file1 = os.path.join(self.train_im_path1,self.train_imgs1[index])
        img1 = Image.open(img_file1)
        img_file2 = os.path.join(self.train_im_path2,self.train_imgs2[index])
        img2 = Image.open(img_file2)
        label_file = str(self.train_labels[index][:-4])+'.png'
        
                 
        im1,im2= self.transform(img1,img2)
        
      
        #lb = lb[0]
      

  
        return im1,im2, label_file

    def transform(self, img1,img2):
        

        transform_img = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        transform_img_2 = transforms.Compose([
                                            #transforms.Resize((384,384)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
                                            ])
        im1 = transform_img_2(img1)
        im2 = transform_img_2(img2)
        #label = transform_img_2(label)
        return im1,im2
 

#a = MyData()
#print(a[0][2][0:3,238:244])
'''
for i in range(256):
    for j in range(256):
        if a[0][2][i,j] == 1:
            print([i,j])
'''
