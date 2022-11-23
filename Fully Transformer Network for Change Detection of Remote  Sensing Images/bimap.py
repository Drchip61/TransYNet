import cv2
import numpy as np
import os
import shutil

refile ='cnn_v2_gz'
outPath = 'cnn_v2_gz_deal'
if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.mkdir(outPath)
name = os.listdir(refile)

for i in range(len(name)):
    label_file = os.path.join(refile,name[i])
    a = cv2.imread(label_file,0)
    
    b = 2.*np.mean(a)
    b,photo = cv2.threshold(a,6,255,cv2.THRESH_BINARY)
    #photo = cv2.resize(photo, (256,256),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(outPath+'/'+name[i],photo)


