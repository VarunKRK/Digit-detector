
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


##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################


def predict_model_PETER(path):

     model = md.DigitRecognizer()
     checkpoint = torch.load('trained_model_PETER.tar')
     model.load_state_dict(checkpoint['model'])
     # print(checkpoint)
     # model = model.load_weights(checkpoint)

     original_image = plt.imread(path)
     processed_image = dh.preprocess_one_digit(path)

     predict_image = torch.from_numpy(processed_image)
     predict_image = predict_image.flatten().float()

     output = model.forward(predict_image)
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


# path_1 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_00.jpeg"
# path_2 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_2.jpeg"
# path_3 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_3.jpeg"
# path_4 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_4.jpeg"
# path_5 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_5.jpeg"
# path_6 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_6.jpeg"
# path_7 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_7.jpeg"
# path_8 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_8.jpeg"
# path_9 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_9.jpeg"

# paths = [path_2, path_3, path_4, path_5, path_6, path_7, path_8, path_9]

# for path in paths:
#      predict_model_PETER(path)



##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################



def predict_model_VARUN(path):

     model = md.DigitRecognizer()
     checkpoint = torch.load('trained_model_VARUN.tar')
     model.load_state_dict(checkpoint['model'])
     # print(checkpoint)
     # model = model.load_weights(checkpoint)

     original_image = plt.imread(path)
     processed_image = dh.preprocess_one_digit(path)

     predict_image = torch.from_numpy(processed_image)
     predict_image = predict_image.flatten().float()

     output = model.forward(predict_image)
     _, pred = torch.max(output, -1)

     fig = plt.figure()
     plt.figure(figsize=(10, 5))

     plt.subplot(1, 2, 1)
     plt.imshow(original_image)
     plt.title(f'Original Image: ', fontsize=20)
     # plt.title(f'Original Image \nPrediction : { pred.item()}', fontsize=20)

     plt.subplot(1, 2, 2)
     plt.imshow(processed_image, cmap='gray')
     plt.title( f'Prediction : { pred.item()}', fontsize=20)

     plt.show()
     return 


# path_1 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_00.jpeg"
# path_2 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_2.jpeg"
# path_3 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_3.jpeg"
# path_4 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_4.jpeg"
# path_5 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_5.jpeg"
# path_6 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_6.jpeg"
# path_7 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_7.jpeg"
# path_8 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_8.jpeg"
# path_9 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_9.jpeg"

# paths = [path_2, path_3, path_4, path_5, path_6, path_7, path_8, path_9]

# for path in paths:
pt = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/"

pth_0 = "digit_0_ALESSIA.jpeg"
pth_1 = "digit_1_ALESSIA.jpeg"
pth_2 = "digit_2_ALESSIA.jpeg"
pth_3 = "digit_3_ALESSIA.jpeg"
pth_4 = "digit_4_ALESSIA.jpeg"
pth_5 = "digit_5_ALESSIA.jpeg"
pth_6 = "digit_6_ALESSIA.jpeg"
pth_7 = "digit_7_ALESSIA.jpeg"
pth_8 = "digit_8_ALESSIA.jpeg"
pth_9 = "digit_9_ALESSIA.jpeg"



# pth_4 = "digit_4_VARUN.jpg"
# pth_5 = "digit_5_VARUN.jpg"
# pth_6 = "digit_6_VARUN.jpg"
# pth_7 = "digit_7_VARUN.jpg"
# pth_8 = "digit_8_VARUN.jpg"
# pth_9 = "digit_9_VARUN.jpg"


# date3 = [pth_0, pth_1, ]

# path = pt + pth_5

for i in range(9):
     path = pt + f'digit_{i}_ALESSIA.jpeg'
     predict_model_VARUN(path)




##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################

