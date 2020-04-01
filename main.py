
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import sys, os, signal
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import darknet
import time, copy, utils
from time import process_time, localtime, strftime
import vars
from train import train_model, test_model, visualize_model
import matplotlib.pyplot as plt
from torchsummary import summary
import mymodels

print(torch.__version__)
# برای تقسیم داده ها به آموزش و تست به صورت خودکار
# این تابع درصدی از داده های آموزش را ضمن حفظ اسامی پوشه ها، به پوشه تست منتقل می کند 
if(not os.path.exists(vars.val_dir)):
    utils.create_validation_data(vars.train_dir, vars.val_dir, vars.val_split_ratio, 'jpg')


def handler(signum, frame):
	print('Signal handler called with signal', signum)
	print('Training will finish after this epoch')
	vars.stop_training = True
	#raise OSError("Couldn't open vars.device!")

signal.signal(signal.SIGINT, handler) # only in python version >= 3.2

print("Start Time: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
print("Active Mode: " + vars.mode)
plt.ion()   # interactive mode
######################################################################
# Load Data
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([vars.input_size, vars.input_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([vars.input_size, vars.input_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize([vars.input_size, vars.input_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# image_dataset_train = {'train': datasets.ImageFolder(os.path.join(vars.data_dir, 'train'), data_transforms['train'])}
# image_dataset_test = {'val': datasets.ImageFolder(os.path.join(vars.data_dir, 'val'), data_transforms['val'])}
# image_dataset_train.update(image_dataset_test)
# خط پایین با سه خط بالا برابری می کند!
image_datasets = {x: datasets.ImageFolder(os.path.join(vars.data_dir, x), data_transforms[x])
                  for x in ['train', 'val', 'test']}

vars.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=vars.batch_size, shuffle=True, num_workers=0)
              for x in ['train', 'val', 'test']}

vars.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
vars.class_names = image_datasets['train'].classes

# Get a batch of training data
inputs, classes = next(iter(vars.dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

#utils.imshow(out, title=[vars.class_names[x] for x in classes])


######################################################################
# Finetuning the convnet
# Load a pretrained model and reset final fully connected layer.

##\\//\\//model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

if(vars.model_name.find('vgg') != -1):
    model = models.vgg11_bn(pretrained=vars.pre_trained)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, len(vars.class_names))
elif(vars.model_name == 'resnet152'):
    model = models.resnet152(pretrained=vars.pre_trained)    
elif(vars.model_name == 'resnet50'):
    model = models.resnet50(pretrained=vars.pre_trained)
elif(vars.model_name == 'resnet18'):
    model = models.resnet18(pretrained=vars.pre_trained)
elif(vars.model_name == 'googlenet'):
    model = models.googlenet(pretrained=vars.pre_trained)
elif(vars.model_name == 'darknet53'):
    model = darknet.darknet53(1000)
    if(vars.pre_trained):
        model.load_state_dict(torch.load('D:\\Projects\\_Python\\Fruit Detection2\\darknet53.weights'))    
elif(vars.model_name == 'fruit3conv'):
    model = mymodels.fruit3conv()
elif(vars.model_name == 'fruit5conv'):
    model = mymodels.fruit5conv()    
elif(vars.model_name == 'fruit2conv'):
    model = mymodels.fruit2conv()        
elif(vars.model_name == 'mymodel'):
    model = mymodels.MyModel()
elif(vars.model_name == 'mymodel2'):
    model = mymodels.MyModel2()


if(vars.model_name != 'vgg11'):
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(vars.class_names))

model.to(vars.device)
print(summary(model, input_size=(3, vars.input_size, vars.input_size), batch_size=-1, device=vars.device.type))

if(vars.mode == 'test'):#test
    #model.load_state_dict(torch.load("D:\\Projects\\Python\\Zeitoon Detection\"))
    model.load_state_dict(torch.load(vars.test_model))
    model = model.to(vars.device)
    
    test_model(model, vars.criterion, 'test')
else:
    optimizer = optim.SGD(model.parameters(), lr=0.001 if vars.pre_trained else 0.06, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.05)
    # Decay LR by a factor of 0.6 every 6 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10 if vars.pre_trained else 6, gamma=0.6)
    model = model.to(vars.device)
    model = train_model(model, vars.criterion, optimizer, exp_lr_scheduler, vars.num_epochs)
    visualize_model(model)

# ######################################################################
# # ConvNet as fixed feature extractor
# # Here, we need to freeze all the network except the final layer. We need
# # to set ``requires_grad == False`` to freeze the parameters so that the
# # gradients are not computed in ``backward()``.
# # You can read more about this in the documentation
# # `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
# #

# model_conv = torchvision.models.resnet18(pretrained=True)
# for param in model_conv.parameters():
#     param.requires_grad = False

# # Parameters of newly constructed modules have requires_grad=True by default
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, len(vars.class_names))

# model_conv = model_conv.to(vars.device)

# criterion = nn.CrossEntropyLoss()

# # Observe that only parameters of final layer are being optimized as
# # opposed to before.
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# ######################################################################
# # Train and evaluate
# # ^^^^^^^^^^^^^^^^^^
# #
# # On CPU this will take about half the time compared to previous scenario.
# # This is expected as gradients don't need to be computed for most of the
# # network. However, forward does need to be computed.
# #

# model_conv = train_model(model_conv, criterion, optimizer_conv,
#                          exp_lr_scheduler, vars.num_epochs=25)

# ######################################################################
# #

# visualize_model(model_conv)

plt.ioff()
plt.show()

######################################################################
# checkout `Quantized Transfer Learning for Computer Vision Tutorial <https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html>`_.
