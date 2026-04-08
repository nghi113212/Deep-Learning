# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:31:57 2022

@author: tuann
"""
from torchsummary import summary
# from models.M1 import *
from models.modelMidterm import *
# from models.modelBT1 import *

# model = CNN(224,5)
model = M1(num_classes=5)

model = model.cpu()
print ("model")
print (model)

# get the number of model parameters
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
#print(model)
#model.cuda()
summary(model, (3, 224, 224))
# summary(model, (3, 32, 32))