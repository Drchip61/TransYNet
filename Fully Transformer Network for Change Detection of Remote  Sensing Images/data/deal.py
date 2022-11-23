from PIL import Image
import os 
import cv2
Files_path = r"test_deal"
labels_num = len(os.listdir(Files_path))
print(labels_num)

for i in range(labels_num):
    image_dir = os.path.join(Files_path,str(os.listdir(Files_path)[i]))
    #image_list = os.listdir(image_dir)
    #for image_name in image_dir:
    image_path = os.path.join(image_dir)
    #img = Image.open(image_path)
    #print(type(img))
    #Resize_img = img.resize((256,256),Image.NEAREST)
    #img = img[:,:,0]
    img = cv2.imread(image_path)
    img = img[:,:,0]
    #lb0 = cv2.resize(img,(256,256),interpolation=cv2.INTER_NEAREST)
    outpath = "try2"
    lb0 = cv2.merge([img*255.])
    cv2.imwrite(outpath+'/'+str(os.listdir(Files_path)[i]),lb0) 

    #Resize_img.save(outpath+'/'+str(os.listdir(Files_path)[i]))


