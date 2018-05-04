import os
import cv2
import numpy as np
from PIL import Image
import numpy as np
import os
import pandas as pd
import torch
import torch.utils.data
import argparse
import time
import csv
import torch.nn as nn
from torch import optim
import math
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from collections import namedtuple
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import time
from utils.LogicRLUtils import *


def tuple_tostring(tuple):
	return ','.join(tuple[1:-2].split(" "))

def parse_txt(full_textname): 
	file_ = open(full_textname)
	parsed_text = list(filter(None, [tuple_tostring(line) for line in file_]))
	file_.close()	
	return parsed_text

def parse_annotation(text_dir, img_dir, label_file):
	# LABELS dict
	labels_ = parse_txt(label_file)
	LABELS = {}
	IDX_TO_LABELSTR = {}
	i = 0
	for label in labels_:
		LABELS[label] = i
		IDX_TO_LABELSTR[i] = label
		i += 1
	CLASS = len(LABELS)

	all_imgs = []
	all_labels = []

	file_names = [name for name in os.listdir(text_dir) if name.endswith('.txt')] # based on those with bboxes

	for file_name in file_names:
		img = {'object':[]}

		full_textname = text_dir + '/' + file_name
		full_imgname = img_dir + '/' + file_name.replace('txt', 'png') # TODO

		img_label = parse_txt(full_textname)

		label_encode = np.zeros((1,CLASS)).astype(int)
		for label in img_label:
			label_encode[0, LABELS[label]] = 1

		im = Image.open(full_imgname)
		image = np.array(im)
		im.close()
		
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		#img_ = Image.fromarray(image, 'L')
		#img_.show()
		
		image = cv2.resize(image, (84, 84), interpolation = cv2.INTER_CUBIC)[np.newaxis,np.newaxis,:,:]
		
		all_labels.append(label_encode)
		all_imgs.append(image)

	all_labels = np.concatenate(all_labels)	
	all_imgs = np.concatenate(all_imgs)	

	return all_imgs, all_labels, LABELS, IDX_TO_LABELSTR, CLASS


class CNNModel(torch.nn.Module):
	'''
	reuse some of my code from hw

	scalable version of state decoder
	could deal with increasing number of states

	input: @CLASSES, number of states

	input of decode_state: greyscale image of shape (1, 1, 84, 84)
	output of decode_state: list of detected states, ['actorInRoom,room_1', 'actorOnSpot,room_1,conveyor_1']

	example: 
	at the beggining, only have 15 different states to detect, so use CLASSES = [15]
	then finding say 3 more states, use CLASSES = [15, 3].

	if more states is added, should call model.model_train() to let the network learn train first.
	'''
	def __init__(self, CLASSES, 
		pretrained_model_pth='./save_weights/parser_epoch_23_loss_0.0002853113460746086_valacc_0.9986979166666667.t7'):
		super(CNNModel, self).__init__()

		all_imgs, all_labels, LABELS, IDX_TO_LABELSTR, CLASS = parse_annotation('lev1_labeled/imgLevel1Label', 'lev1_labeled/imgLevel1Label', 'lev1_labeled/0_allpossible.txt')

		X_train, X_valid, y_train, y_valid = train_test_split(
				all_imgs, all_labels, random_state=6060, train_size=0.75)

		args = namedtuple('args',
						  [
							  'batch_size',
							  'save_directory',
							  'epochs',
							  'init_lr',
							  'cuda'])(
			32,
			'save_weights/',
			40,
			1e-4,
			False)

		kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
		train_loader = DataLoader(
			myDataset(X_train, y_train), shuffle=True,
			batch_size=args.batch_size, **kwargs)
		valid_loader = DataLoader(
			myDataset(X_valid, y_valid), shuffle=True,
			batch_size=args.batch_size, **kwargs)


		# graph
		dropout = 0.5
		self.dropout = dropout
		self.IDX_TO_LABELSTR = IDX_TO_LABELSTR
		self.CLASSES = CLASSES

		# input (1, 84, 84)
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
		self.relu1 = nn.ReLU()
		#self.bn1 = nn.BatchNorm1d(256)
		#self.drop1 = nn.Dropout(dropout)

		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
		self.relu2 = nn.ReLU()
		#self.bn2 = nn.BatchNorm1d(384)
		#self.drop2 = nn.Dropout(dropout)

		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
		self.relu3 = nn.ReLU()
		#self.bn3 = nn.BatchNorm1d(512)
		#self.drop3 = nn.Dropout(dropout)

		# flatten

		self.linear1 = nn.Linear(in_features=3136,
									out_features=256)
		self.relulinear1 = nn.ReLU()
		#self.droplinear1 = nn.Dropout(0.15)

		for i, class_num in enumerate(CLASSES):
			setattr(self, 'projection_{}'.format(i), nn.Linear(in_features=256,
										out_features=class_num))
			#self.projection = nn.Linear(in_features=256,
			#                            out_features=CLASSES[0])
		self.sigmoid = nn.Sigmoid()

		# initialization
		#self.apply(wsj_initializer)

		self.args = args
		self.dataloader = train_loader
		self.valid_dataloader = valid_loader
		self.criterion = nn.BCELoss()
		self.best_validation_acc = 0
		self.model_param_str = 'weights'

		self.optimizer = optim.Adam(self.parameters(), lr=args.init_lr)
		#self.optimizer = optim.RMSprop(self.parameters(), lr=args.init_lr)
		#self.optimizer = optim.SGD(self.parameters(), lr=args.init_lr)

		# load pretrained weights
		pretrained_dict = torch.load(pretrained_model_pth)
		model_dict = self.state_dict()

		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict) 
		self.load_state_dict(model_dict)

		#self.load_state_dict(pretrained_dict)


		if torch.cuda.is_available():
			self.cuda()

	def forward(self, x):
		h = self.relu1(self.conv1(x))
		h = self.relu2(self.conv2(h))
		h = self.relu3(self.conv3(h))

		# flatten
		h_size = h.size()
		h = h.view(h_size[0], -1)

		h = self.relulinear1(self.linear1(h))

		#output = self.sigmoid(self.projection(h))
		outputs = []
		for i in range(len(self.CLASSES)):
			outputs.append(self.sigmoid(getattr(self, 'projection_{}'.format(i))(h)))
			
		outputs = torch.cat(outputs, 1)
		
		return outputs

	def model_train(self):
		for i in range(self.args.epochs):
			print("---------epoch {}---------".format(i))
			start_time = time.time()
			self.train()  # right place

			losses = 0
			total_cnt = 0

			for input_x, label in self.dataloader:
				self.zero_grad()

				output = self.forward(to_variable(input_x)) 

				loss = self.criterion(output, to_variable(label))

				total_cnt += 1
				losses += loss.data[0]

				loss.backward()
				self.optimizer.step()
			print("training loss: {}".format(losses / total_cnt / self.args.batch_size))
			validation_acc = self.evaluate()
			if validation_acc > self.best_validation_acc:
				print("--------saving best model--------")
				if not os.path.exists(self.args.save_directory):
					os.makedirs(self.args.save_directory)
				self.model_param_str = \
					'{}parser_epoch_{}_loss_{}_valacc_{}'.format(
						self.args.save_directory, i, losses / total_cnt / self.args.batch_size, validation_acc)
				torch.save(self.state_dict(), self.model_param_str + '.t7')
				self.best_validation_acc = validation_acc

			print("--- %s seconds ---" % (time.time() - start_time))	

		return self.model_param_str		

	def evaluate(self):
		self.eval()

		losses = 0
		total_cnt = 0
		validation_accs = []
		for input_x, label in self.valid_dataloader:
			total_cnt += 1
			output = self.forward(to_variable(input_x))

			loss = self.criterion(output, to_variable(label))
			losses += loss.data[0]
			
			output = output.data
			cond1 = output < 0.5
			cond2 = output >= 0.5
			output[cond1] = 0
			output[cond2] = 1
			shape = output.shape
			#print("shape:", shape) # (B, Class_num)
			#print("output", output)
			#print("label", label)
			validation_accs.append(torch.sum(output == label) / shape[0] / shape[1])

		losses /= total_cnt * self.args.batch_size
		print("validation loss: {}".format(losses))
		validation_acc = np.mean(validation_accs)
		print("validation accuracy: {}".format(validation_acc))
		return validation_acc
	
	def decode_state(self, input_x):
		'''input of shape (1, 1, 84, 84)'''
		self.eval()
		
		output = self.forward(to_variable(input_x))

		output = output.data.numpy()[0] # TODO
		decoded_state = get_state(output, self.IDX_TO_LABELSTR)
		
		return decoded_state

	def decode_state_logits(self, input_x):
		'''input of shape (1, 1, 84, 84)'''
		self.eval()
		
		output = self.forward(to_variable(input_x))

		output = output.data.numpy()[0] # TODO
		decoded_state = get_state_logits(output, self.IDX_TO_LABELSTR)
		
		return decoded_state

def get_state_logits(img_label, IDX_TO_LABELSTR):
	labels = []
	for i, logit in enumerate(img_label):
		labels.append((IDX_TO_LABELSTR[i], logit))
	return labels

def get_state(img_label, IDX_TO_LABELSTR):
	labels = []
	for i, logit in enumerate(img_label):
		if logit >= 0.5:
			labels.append(IDX_TO_LABELSTR[i])
	return labels

def to_tensor(numpy_array, datatype):
	# Numpy array -> Tensor
	if datatype == 'int':
		return torch.from_numpy(numpy_array).int()
	elif datatype == 'long':
		return torch.from_numpy(numpy_array).long()
	else:
		return torch.from_numpy(numpy_array).float()


def to_variable(tensor, cpu=False):
	# Tensor -> Variable (on GPU if possible)
	if torch.cuda.is_available() and not cpu:
		# Tensor -> GPU Tensor
		tensor = tensor.cuda()
	return torch.autograd.Variable(tensor)

class myDataset(torch.utils.data.Dataset):
	def __init__(self, input_x, labels, test=False):
		self.input_x = torch.from_numpy(input_x).float()
		self.labels = torch.from_numpy(labels).float()

	def __getitem__(self, index):
		return self.input_x[index], self.labels[index]

	def __len__(self):
		return len(self.input_x)

def main():
	CLASSES = [15]
	model = CNNModel(CLASSES)
	#model.model_train()
	#state = torch.Tensor(1, 1, 84, 84)
	state = FrameToDecoderState(np.load('errimg_0.npy'))
	#decoded_state = model.decode_state(state)
	decoded_state = model.decode_state_logits(state)
	print("decoded state: {}".format(decoded_state))
	#print(model)


if __name__ == '__main__':
	main()
