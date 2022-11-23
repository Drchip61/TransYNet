#from PIL import Image
import cv2
import os 
#Files_path = r"label_256"

#print(labels_num)
import numpy as np
#gt_path = 'data/test/try'
gt_path = 'data/WHU-CD/dataset224/test/label'#'data/CD_Data_GZ/test_224/label'#'data/WHU-CD/dataset224/test/label'#'data/WHU-CD/whu_test256/label'#'data/test/label_Split'#'data/CD_Data_GZ/GZtest256/label'##'data/CD_Data_GZ/test_224/label'#'data/WHU-CD/dataset224/test/label'#'data/SYSU224/test/label_thresh'# #'data/SYSU224/test/label_thresh'
pred_path = 'try/'
labels_num = len(os.listdir(gt_path))

print(labels_num)
#img1 = cv2.imread('test_20.png',flags=0)
#img2 = cv2.imread('test_20.png',flags=0)
def P_R_IoU(gt,pred):

    predict_precision = 0
    predict_recall = 0
    tp = 0
    tn = 0
    for k in range(len(gt)):

        img1 = gt[k]
        img2 = pred[k]
        #print(img1.shape,img2.shape)
        for i in range(224):
            for j in range(224):
                if not(int(img1[i,j]) - int(img2[i,j])) and img2[i,j] == 255:
                    tp += 1  #TP value
                if img1[i,j] == img2[i,j] and img2[i,j] == 0:
                    tn += 1 #TN value
                '''if img1[i,j] ==0 and img2[i,j] == 255: #FP value
                    fp += 1
                if img1[i,j] ==255 and img2[i,j] == 0: #FP value
                    fn += 1  '''

        predict_precision += np.sum(np.reshape(img2,(img2.size,)))/255
        predict_recall += np.sum(np.reshape(img1, (img1.size,))) / 255
        print(k)
    predict_iou = predict_precision+predict_recall-tp
    return tp/predict_precision, tp/predict_recall, tp/predict_iou, (tn+tp)/(predict_iou+tn)




def f1_score(precision, recall):
    return 2*precision*recall/(precision+recall)
           
#print(precision(img1,img2))
def get_average(list_):
    sum_ = 0
    for _ in list_:
        sum_ += _
    return sum_/len(list_)

gt_list = []
pred_list = []

for i in range(labels_num):
    gt = os.path.join(gt_path,str(os.listdir(gt_path)[i]))
    pred = os.path.join(pred_path,str(os.listdir(pred_path)[i]))
    #image_list = os.listdir(image_dir)
    #for image_name in image_dir:
    gt_path1 = os.path.join(gt)
    pred_path1 = os.path.join(pred)
    gt1 = cv2.imread(gt_path1,flags=0)
    pred1 = cv2.imread(pred_path1,flags=0)
    #gt1 = cv2.resize(gt1,(256,256),interpolation = cv2.INTER_NEAREST)
    #pred1 = cv2.resize(pred1,(256,256),interpolation = cv2.INTER_NEAREST)
    gt_list.append(gt1)
    pred_list.append(pred1)

    

precision_res, recall_res, iou_res, OA = P_R_IoU(gt_list,pred_list)

f1_score = f1_score(precision_res,recall_res)
print('precision:',precision_res,'recall:',recall_res,'\n','f1:',f1_score,'IoU:',iou_res,'OA:',OA)


