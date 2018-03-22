#! /usr/bin/env python

"""
This script takes in a configuration file and produces the best model. 
The configuration file is a json file and looks like this:
{
	"model" : {
		"architecture":         "VGG16",
		"input_size":           224,
		"anchors":              [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
		"max_box_per_image":    10,        
		"labels":               ['joe','ladder','skull','key','door','belt','rope']
	},
	"train": {
		"train_image_folder":   '/Users/sw/programming/10703/project/test_images/train_image_folder',
		"train_annot_folder":   '/Users/sw/programming/10703/project/test_images/train_annot_folder',      
		  
		"train_times":          10,
		"pretrained_weights":   "vgg16_weights.h5",
		"batch_size":           2,
		"learning_rate":        1e-4,
		"nb_epoch":             50,
		"warmup_epochs":        3,
		"object_scale":         5.0 ,
		"no_object_scale":      1.0,
		"coord_scale":          1.0,
		"class_scale":          1.0,
		"debug":                true
	},
	"valid": {
		"valid_image_folder":   '/Users/sw/programming/10703/project/test_images/valid_image_folder',
		"valid_annot_folder":   '/Users/sw/programming/10703/project/test_images/valid_annot_folder',
		"valid_times":          1
	}
}
"""

from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import numpy as np
import pickle
import os, cv2
from preprocessing import parse_annotation, BatchGenerator
from utils import WeightReader, decode_netout, draw_boxes, normalize
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import os
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
	description='Train and validate VGG16 model on any dataset')

argparser.add_argument(
	'-c',
	'--conf',
	help='path to configuration file')

argparser.add_argument('-t', '--training', default=False, type=lambda x: (str(x).lower() == 'true'))

def _main_(args):

	config_path = args.conf

	with open(config_path) as config_buffer:    
		config = json.loads(config_buffer.read())

	###############################
	#   Parse the annotations 
	###############################

	# parse annotations of the training set
	train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'], 
												config['train']['train_image_folder'], 
												config['model']['labels'])

	# parse annotations of the validation set, if any, otherwise split the training set
	if os.path.exists(config['valid']['valid_annot_folder']):
		valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'], 
													config['valid']['valid_image_folder'], 
													config['model']['labels'])
	else:
		train_valid_split = int(0.8*len(train_imgs))
		np.random.shuffle(train_imgs)

		valid_imgs = train_imgs[train_valid_split:]
		train_imgs = train_imgs[:train_valid_split]

	if len(config['model']['labels']) > 0:
		overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

		print 'Seen labels:\t', train_labels
		print 'Given labels:\t', config['model']['labels']
		print 'Overlap labels:\t', overlap_labels           

		if len(overlap_labels) < len(config['model']['labels']):
			print 'Some labels have no annotations! Please revise the list of labels in the config.json file!'
			return
	else:
		print 'No labels are provided. Train on all seen labels.'
		config['model']['labels'] = train_labels.keys()
		
	###############################
	#   Construct the model 
	###############################

	yolo = YOLO(architecture        = config['model']['architecture'],
				input_size          = config['model']['input_size'], 
				labels              = config['model']['labels'], 
				max_box_per_image   = config['model']['max_box_per_image'],
				anchors             = config['model']['anchors'])

	###############################
	#   Load the pretrained weights (if any) 
	###############################    

	if os.path.exists(config['train']['pretrained_weights']):
		print "Loading pre-trained weights in", config['train']['pretrained_weights']
		yolo.load_weights(config['train']['pretrained_weights'])

	for layer in galaxyModel.layers:
		print(layer)
        layer.trainable = True	

	###############################
	#   Start the training process 
	###############################

	if args.training:
		yolo.train(train_imgs         = train_imgs,
				   valid_imgs         = valid_imgs,
				   train_times        = config['train']['train_times'],
				   valid_times        = config['valid']['valid_times'],
				   nb_epoch           = config['train']['nb_epoch'], 
				   learning_rate      = config['train']['learning_rate'], 
				   batch_size         = config['train']['batch_size'],
				   warmup_epochs      = config['train']['warmup_epochs'],
				   object_scale       = config['train']['object_scale'],
				   no_object_scale    = config['train']['no_object_scale'],
				   coord_scale        = config['train']['coord_scale'],
				   class_scale        = config['train']['class_scale'],
				   saved_weights_name = config['train']['saved_weights_name'],
				   debug              = config['train']['debug'])
	
	image = cv2.imread(config['valid']['valid_image_folder'] + '/10.png')

	plt.figure(figsize=(10,10))

	boxes = yolo.predict(image)

	image = draw_boxes(image, boxes, labels=config['model']['labels'])

	plt.imshow(image[:,:,::-1]); plt.show()

if __name__ == '__main__':
	# python vgg16.py --conf=vgg16.json
	args = argparser.parse_args()
	_main_(args)
