import cv2
import os
import shutil

outpath = 'label_Split_384'
if os.path.exists(outpath):
    shutil.rmtree(outpath)
os.mkdir(outpath)

size = []
for i in range(3):
    for j in range(3):
        size.append(((384-64)*i,(384-64)*j,384*(i+1)-64*i,384*(j+1)-64*j))
print(size)

def split(img,img_name):
    t=0
    for i in size:
        t+=1
        (x0,y0,x1,y1)=(i[0],i[1],i[2],i[3])
        crop_img = img[x0:x1,y0:y1]
        cv2.imwrite(outpath+'/'+img_name[6:-4]+'_'+str(t)+'.png',crop_img)
        print(outpath+'/'+img_name[6:-4]+'_'+str(t)+'.png')


Files_path = r"label"
labels_num = len(os.listdir(Files_path))
print(labels_num)

for i in range(labels_num):
    
    image_dir = os.path.join(Files_path,str(os.listdir(Files_path)[i]))
    image_path = os.path.join(image_dir)
    img = cv2.imread(image_path)
    split(img,image_path)
    #break

