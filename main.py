# -*- coding: utf-8 -*-
import glob
from tqdm import tqdm
import os
import cv2 as cv
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from DustNet import *
from DustDataLoader import *

gpus = [3]  # 使用哪几个GPU进行训练，这里选择0号GPU
cuda_gpu = torch.cuda.is_available()  # 判断GPU是否存在可用
device = torch.device('cuda:3')
print('cuda_gpu:', cuda_gpu)
print(device)


real_path = os.path.realpath(__file__)
real_dir = real_path[:real_path.rfind('/')]
data_dir = os.path.join(real_dir, 'data')
print(data_dir)

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


classes = ('low dust', 'medium dust', 'high dust')

net = DustNet()
if(cuda_gpu):
    net.to(device)

crossEntropyLoss = nn.CrossEntropyLoss()
mseLoss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=0.0001)
# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0)

for epoch in tqdm(range(100)):  # loop over the dataset multiple times

    running_loss = 0.0
    i = 0
    correct = 0
    total = 0
    for data in dust_train_dataloader:
        i += 1
        # get the inputs

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output_class = net(inputs)

        a = np.array(np.argmax(output_class.cpu().detach().numpy(), axis=1))
        b = np.array(labels.cpu().detach().numpy())
        correct += np.sum(a == b)
        total += len(labels)
        # print('labels.shape', labels.shape)
        # print('output_class.shape', output_class.shape)
        loss = crossEntropyLoss(output_class, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i == 10:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i, running_loss / 10))
            print(f'i: {i} acc: {correct / total}')
            correct = 0
            total = 0
            running_loss = 0.0
            i = 1
            with torch.no_grad():
                acc = [0, 0]
                for data in dust_valid_dataloader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    output_class = net(inputs)
                    output_class = output_class.cpu()
                    a = np.array(np.argmax(output_class, axis=1))
                    b = np.array(labels.cpu())
                    # print(a.shape, b.shape)
                    acc[0] += np.sum(a == b)
                    acc[1] += len(labels)
                print(f'valid acc: {acc[0]/acc[1]}')


print('Finished Training')

# ########################################################################
# # 5. Test the network on the test data
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# #
# # We have trained the network for 2 passes over the training dataset.
# # But we need to check if the network has learnt anything at all.
# #
# # We will check this by predicting the class label that the neural network
# # outputs, and checking it against the ground-truth. If the prediction is
# # correct, we add the sample to the list of correct predictions.
# #
# # Okay, first step. Let us display an image from the test set to get familiar.

# dataiter = iter(testloader)
# images, labels = dataiter.next()

# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# ########################################################################
# # Okay, now let us see what the neural network thinks these examples above are:

# outputs = net(images)

# ########################################################################
# # The outputs are energies for the 10 classes.
# # Higher the energy for a class, the more the network
# # thinks that the image is of the particular class.
# # So, let's get the index of the highest energy:
# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))

# ########################################################################
# # The results seem pretty good.
# #
# # Let us look at how the network performs on the whole dataset.

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#         100 * correct / total))

# ########################################################################
# # That looks waaay better than chance, which is 10% accuracy (randomly picking
# # a class out of 10 classes).
# # Seems like the network learnt something.
# #
# # Hmmm, what are the classes that performed well, and the classes that did
# # not perform well:

# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1

# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print(device)
