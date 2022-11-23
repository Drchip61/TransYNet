import cv2
import numpy as np
import os

refile = 'test_res'
name = os.listdir('test_res')
for i in range(len(name)):
    label_file = os.path.join(refile,name[i])
    a = cv2.imread(label_file,0)
    b = 2.*np.mean(a)


    b,photo = cv2.threshold(a,25,255,cv2.THRESH_BINARY)
    cv2.imwrite('try1/'+name[i],photo)


