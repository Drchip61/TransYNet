from PIL import Image
import os 
Files_path = r"try"
labels_num = len(os.listdir(Files_path))
print(labels_num)
import numpy as np

for i in range(labels_num):
    image_dir = os.path.join(Files_path,str(os.listdir(Files_path)[i]))
    #image_list = os.listdir(image_dir)
    #for image_name in image_dir:
    image_path = os.path.join(image_dir)
    img = Image.open(image_path)
    '''
    for i in range(np.array(img).shape[0]):
        for j in range(np.array(img).shape[1]):
            if int(np.array(img)[i,j]) != 0 or int(np.array(img)[i,j]) != 255:
                print(np.array(img)[i,j]) 
    '''
    print(np.array(img).shape) 
    if np.array(img).sum()%5 != 0:
        print(str(os.listdir(Files_path)[i]))
    
    #Resize_img = img.resize((256,256))
    #outpath = "label_256"
    #Resize_img.save(outpath+'/'+str(os.listdir(Files_path)[i]))


