import torch
import torch.nn as nn


device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
pre_trained = False
num_epochs = 25 if pre_trained else 50
stop_training = False
dataloaders = None
dataset_sizes = None
class_names = None
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
data_dir = 'E:/Dataset/Fruits'
train_dir = data_dir + '/train/'
test_dir = data_dir + '/test/'
val_dir = data_dir + '/val/'
val_split_ratio = 0.1

# 'fruit3conv' 'fruit2conv' 'fruit5conv' 'mymodel' 'mymodel2' 'vgg11' 'resnet18' 'resnet50' 'resnet152' 'darknet53' (backbone of yolov3)
model_name = 'vgg11'
input_size = 224 if model_name.find('vgg') != -1 else 256

mode = 'test' #train or test
#test_model = "\\mymodel-sgd-cuda-batch-64\\LR from 0.06 to 0.001\\ep46-acc99.82-loss0.0108.pth"
#test_model = "\\ResNet-SGD-cuda-batch-64\\ep22-acc99.96-loss0.0026.pth"
test_model = "D:\\Projects\\_Python\\Fruit Detection2\\vgg11-SGD-cuda-batch-16-pretrained\\ep2-acc100.00-loss0.0011.pth"