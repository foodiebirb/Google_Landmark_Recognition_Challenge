from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from skimage import io,transform
from torch.utils.data import Dataset, DataLoader
from skimage.viewer import ImageViewer
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

#Constants
ROOT_DIR = 'images/'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 1
BATCH_SIZE = 4
NUM_WORKERS = 4

#Datasets for training and prediction
class LandmarkDataset(Dataset):
	def __init__(self, csv, root_dir, is_train, transform):
		self.df = pd.read_csv(csv)
		self.root_dir = root_dir
		self.is_train = is_train
		self.transform = transform

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		id_col = 1 #shared by both train.csv and test.csv
		label_col = 3 #only train.csv has labels
		label = self.df.iloc[idx, label_col] if self.is_train else [None]
		img_name = (os.path.join(self.root_dir, str(self.df.iloc[idx, id_col])) + '.jpg')
		img = self.transform(io.imread(img_name))
		sample = {'image': img, 'label': label}
		return sample

#Transform images to match the format of ImageNet
def construct_transforms():
	ct = transforms.Compose([
	transforms.ToPILImage(),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	return ct

#Create and format train set and test set
ct = construct_transforms()
trainset = LandmarkDataset(csv = TRAIN_FILE, root_dir = ROOT_DIR, is_train = True, transform = ct)
testset = LandmarkDataset(csv = TEST_FILE, root_dir = ROOT_DIR, is_train = False, transform = ct)
trainloaders = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloaders = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle = False, num_workers=NUM_WORKERS)

#Finetune
def train_model(model, criterion, optimizer, scheduler):
	scheduler.step()
	model.train()

	total_num = len(trainset)
	running_loss = 0.0
	running_corrects = 0

	print('THE EPOCH STARTS!') 
	print('-' * 10)
	for sample in trainloaders:
		inputs = sample['image'].to(DEVICE)
		labels = sample['label'].to(DEVICE)
		optimizer.zero_grad()
		with torch.set_grad_enabled(True):
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

		running_loss += loss.item() * inputs.size(0)
		running_corrects += torch.sum(preds == labels.data)
		print('Loss: {:.4f}'.format(loss.item()))

		epoch_loss = running_loss/total_num
		epoch_acc = running_corrects.double()/total_num

	print('THE EPOCH ENDS!')
	return model

#Import pretrained resnet18
model_ft = models.resnet18(pretrained=True)

#Set up parameters for the model
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 8)
model_ft = model_ft.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#Finetune the model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
torch.save(model_ft.state_dict(), 'Final_Model')

#Predict on test dataset
def predict(model_name):
	model_ft.load_state_dict(torch.load(model_name))
	model_ft.eval()

	predictions = []
	for sample in testloaders:
		with torch.no_grad():
			inputs = sample['image'].to(DEVICE)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			formatted_preds = preds.numpy()
			for pred in np.nditer(formatted_preds):
				predictions.append(pred)

	return predictions

predictions = predict('Final_Model')

#Save the predictions
def write(predictions):
	file = open("submission.txt","w+")
	file.write("landmatk_id\n")
	for pred in predictions:
		file.write('%s\n' %pred)
	file.close()

write(predictions)

