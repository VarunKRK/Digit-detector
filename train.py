import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import Classifier
from data_handler import trainloader, testloader




model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)


epochs = 10

train_loss = []
test_loss = []

print_every = 40

for epoch in range(epochs):
    running_loss_train = 0
    

    for images, labels in trainloader:

        
                       
        optimizer.zero_grad()               ### Step-1 in training loop - reset the gradients

        pred = model.forward(images)      ### step-2 forward pass

        loss_train = criterion(pred, labels) ### step-3 Compute loss

        loss_train.backward()                ### step-4 Backward pass

        optimizer.step()                     ### step-5 update the model

        running_loss_train += loss_train.item()

    else:
        running_loss_test = 0
        
        with torch.no_grad():

            model.eval()
            
            for imgs, labls in testloader:

                

                pred = model.forward(imgs)

                loss_test = criterion(pred, labls)

                running_loss_test += loss_test.item()

    model.train()

    print(f'Epoch: {epoch + 1} | Train loss: {running_loss_train/len(trainloader)} | Test loss: {running_loss_test/len(testloader)}' )

    train_loss.append(running_loss_train/len(trainloader))
    test_loss.append(running_loss_test/len(testloader))

save_path = './cls.pth'
torch.save(model.state_dict(), save_path)

plt.plot(train_loss, label = 'Trainloss')
plt.plot(test_loss, label = 'Testloss')

plt.plot()
plt.legend()
plt.show()



       