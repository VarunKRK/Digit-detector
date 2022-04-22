# print('Welocme, Peter Zorve')
# print('Alessio told me to do it')

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms 


######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

path = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_5.jpeg"


def preprocess_one_digit(path):
     initial_image = cv.imread(path)
     gray_image    = cv.cvtColor(initial_image, cv.COLOR_BGR2GRAY)

     _, threshold_image = cv.threshold(gray_image, 80, 255, cv.THRESH_BINARY_INV)

     coutours_image, associate = cv.findContours(threshold_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

     sorted_contous = sorted(coutours_image, key=cv.contourArea, reverse=True)

     x, y, w, h = cv.boundingRect(sorted_contous[0]) 

     boundary_list = [x, y, (gray_image.shape[0] - (x+w)), (gray_image.shape[0] - (y+h))]
     num = int(min(boundary_list)/2)

     cropped_image = threshold_image.copy()
     cropped_image = cropped_image[y-num : y+h+num,  x-num : x+w+num]

     kernel = np.ones((3, 3))
     dilated_image = cv.dilate(cropped_image, kernel, iterations = 15)

     resized_image = cv.resize(dilated_image, (28, 28))

     return resized_image



# resized_image = preprocess_one_digit(path)
# plt.imshow(resized_image)
# plt.show()



######################################################################################################################################
######################################################################################################################################
######################################################################################################################################




######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

# path2 = "C:/Users/Omistaja/Desktop/deep_learning/hand_written_digits/digit_multi.jpeg"

# length = 5 

# initial_image = cv.imread(path2)
# gray_image    = cv.cvtColor(initial_image, cv.COLOR_BGR2GRAY)

# _, threshold_image = cv.threshold(gray_image, 150, 255, cv.THRESH_BINARY_INV)

# coutours_image, _ = cv.findContours(threshold_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# print('first : ', len(coutours_image))


# # draw_contours = cv.drawContours(threshold_image, coutours_image, -1, (255, 0, 0), 10)

# sorted_contous = sorted(coutours_image, key=cv.contourArea, reverse=True)
# print('2nd', len(sorted_contous))

# sorted_contous = sorted_contous[ :length]

# print('3nd', len(sorted_contous))

# all_images = []

# for i in range(length):
#      x, y, w, h = cv.boundingRect(sorted_contous[i])

#      boundary_list = [x, y, (sorted_contous[i].shape[0] - (x+w)), (sorted_contous[i].shape[0] - (y+h))]
#      num = int(min(boundary_list)/2)

#      cropped_image = threshold_image.copy()
#      cropped_image = cropped_image[y-num : y+h+num,  x-num : x+w+num]

#      # kernel = np.ones((3, 3))
#      # dilated_image = cv.dilate(cropped_image, kernel, iterations = 10)

#      # resized_image = cv.resize(dilated_image, (28, 28))

#      all_images.append(cropped_image)


# plt.imshow(all_images[3])

# plt.show()




# fig = plt.figure()
# plt.figure(figsize=(15, 10))

# plt.subplot(2, 4, 1)
# plt.imshow(initial_image)

# plt.subplot(2, 4, 2)
# plt.imshow(gray_image, cmap='gray')

# plt.subplot(2, 4, 3)
# plt.imshow(threshold_image, cmap='gray')

# plt.subplot(2, 4, 4)
# plt.imshow(draw_contours, cmap='gray')

# # plt.subplot(2, 4, 5)
# # plt.imshow(resized_image, cmap='gray')

