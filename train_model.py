
import torch 
import torch.nn as nn 
import torchvision
import matplotlib.pyplot as plt 
import torch.nn.functional as F
import torchvision.transforms as transforms 
import torchvision.datasets as datasets
import cv2
import models as md 
import data_handler as dh

train_transforms = transforms.Compose([ transforms.Resize((28, 28)),  transforms.Grayscale(),   transforms.ToTensor(), transforms.Normalize([0.5], [0.5])    ])
test_transforms  = transforms.Compose([ transforms.Resize((28, 28)),  transforms.ToTensor(),    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])       ])

root_path = 'C:/Users/Omistaja/Desktop/deep_learning/MNIST/'

train_data = datasets.ImageFolder(root_path + 'train', transform=train_transforms)
test_data  = datasets.ImageFolder(root_path + 'test',  transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=50, shuffle=False)

image, label = next(iter(train_loader))

# print(image.shape, label.shape)
# for i in range(10):
#      # print(label[i])
#      plt.subplot(2, 5, i+1)
#      plt.imshow(image[i][0], cmap='gray')
# plt.show()


##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################


epochs = 10
batch_size = 50

model = md.DigitRecognizer()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

""" Training the Model """
for epoch in range(epochs):
     for step, (images, labels) in enumerate(train_loader):
          
          images = images.reshape(-1, 28*28)

          optimizer.zero_grad()
          predict = model.forward(images)
          loss_train = criterion(predict, labels)
          loss_train.backward()
          optimizer.step()
          print(f'Epoch : {epoch+1}/{epochs}, loss : {loss_train.item()}')


""" Saving the Model """
state = { 'epoch'     : epoch + 1,   
          'model'     : model.state_dict(),
          'optimizer' : optimizer.state_dict()
        }

torch.save(state, 'trained_model.tar')


##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################




##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################