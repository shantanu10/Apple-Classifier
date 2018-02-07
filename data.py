import os
import numpy as np
import cv2

def load_data():

    #loading training data
    list_of_imgs = []
    img_dir = "D:/Projects/Apple Classifier/training_data/apples/"
    for img in os.listdir(img_dir):
        img = os.path.join(img_dir, img)
        if not img.endswith(".jpg"):
            continue
        a = cv2.imread(img).astype(np.float32)
        a = cv2.resize(a,(28,28),0,0, cv2.INTER_LINEAR)
        if a is None:
            print ("Unable to read image", img)
            continue
        list_of_imgs.append(a.flatten())
    train_data = np.asarray(list_of_imgs)
    train_labels = np.full(train_data.shape[0],0,dtype = np.int32)

    list_of_imgs = []
    img_dir = "D:/Projects/Apple Classifier/training_data/others/"
    for img in os.listdir(img_dir):
        img = os.path.join(img_dir, img)
        if not img.endswith(".jpg"):
            continue
        a = cv2.imread(img).astype(np.float32)
        a = cv2.resize(a,(28,28),0,0, cv2.INTER_LINEAR)
        if a is None:
            print ("Unable to read image", img)
            continue
        list_of_imgs.append(a.flatten())
    train_data2 = np.asarray(list_of_imgs)
    train_labels2 = np.full(train_data2.shape[0],1,dtype = np.int32)
    train_data = np.concatenate((train_data,train_data2),axis=0)
    train_labels = np.concatenate((train_labels,train_labels2),axis=0)

    #loading evaluation data

    list_of_imgs = []
    img_dir = "D:/Projects/Apple Classifier/validation_data/apples/"
    for img in os.listdir(img_dir):
        img = os.path.join(img_dir, img)
        if not img.endswith(".jpg"):
            continue
        a = cv2.imread(img).astype(np.float32)
        a = cv2.resize(a,(28,28),0,0, cv2.INTER_LINEAR)
        if a is None:
            print ("Unable to read image", img)
            continue
        list_of_imgs.append(a.flatten())
    eval_data = np.asarray(list_of_imgs)
    eval_labels = np.full(eval_data.shape[0],0,dtype = np.int32)

    list_of_imgs = []
    img_dir = "D:/Projects/Apple Classifier/validation_data/others/"
    for img in os.listdir(img_dir):
        img = os.path.join(img_dir, img)
        if not img.endswith(".jpg"):
            continue
        a = cv2.imread(img).astype(np.float32)
        a = cv2.resize(a,(28,28),0,0, cv2.INTER_LINEAR)
        if a is None:
            print ("Unable to read image", img)
            continue
        list_of_imgs.append(a.flatten())
    eval_data2 = np.asarray(list_of_imgs)
    eval_labels2 = np.full(eval_data2.shape[0],1,dtype = np.int32)
    eval_data = np.concatenate((eval_data,eval_data2),axis=0)
    eval_labels = np.concatenate((eval_labels,eval_labels2),axis=0)
    return (train_data,train_labels,eval_data,eval_labels)


