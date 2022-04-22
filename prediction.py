
import torch 
import torch.nn as nn 
import torchvision
import matplotlib.pyplot as plt 
import torch.nn.functional as F
import torchvision.transforms as transforms 
import torchvision.datasets as datasets
import cv2
# import train_model as td

import models as md 
import data_handler as dh


""" Load the Model """
checkpoint = torch.load('trained_model.tar')

model = md.DigitRecognizer()

def predict(path):
     original_image = plt.imread(path)
     processed_image = dh.preprocess_one_digit(path)

     predict_image = torch.from_numpy(processed_image)
     predict_image = predict_image.flatten().float()

     output = model(predict_image)
     _, pred = torch.max(output, -1)

     fig = plt.figure()
     plt.figure(figsize=(10, 5))

     plt.subplot(1, 2, 1)
     plt.imshow(original_image)
     plt.title('Original Image', fontsize=20)

     plt.subplot(1, 2, 2)
     plt.imshow(processed_image, cmap='gray')
     plt.title( f'Prediction : { pred.item()}', fontsize=20)

     plt.show()
     return 


path_1 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_00.jpeg"
path_2 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_2.jpeg"
path_3 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_3.jpeg"
path_4 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_4.jpeg"
path_5 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_5.jpeg"
path_6 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_6.jpeg"
path_7 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_7.jpeg"
path_8 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_8.jpeg"
path_9 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_9.jpeg"

paths = [path_1, path_2, path_3, path_4, path_5, path_6, path_7, path_8, path_9]

for path in paths:
     predict(path)