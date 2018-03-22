import os
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop, Adam
from PIL import Image
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model


def tuple_tostring(tuple):
	return ','.join(tuple[1:-2].split(" "))

def parse_txt(full_textname): 
	file_ = open(full_textname)
	parsed_text = list(filter(None, [tuple_tostring(line) for line in file_]))
	file_.close()	
	return parsed_text

def parse_annotation(text_dir, img_dir, label_file):
	'''
	new parse_annotation code
	example use:
	text_dir = '/Users/sw/programming/10703/project/yolo-boundingbox-labeler-GUI/bbox_txt'
	img_dir = '/Users/sw/programming/10703/project/yolo-boundingbox-labeler-GUI/images'
	'''
	# LABELS dict
	labels_ = parse_txt(label_file)
	LABELS = {}
	i = 0
	for label in labels_:
		LABELS[label] = i
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
		image = cv2.resize(image, (84, 84), interpolation = cv2.INTER_CUBIC)[np.newaxis,:,:,np.newaxis]
		
		all_labels.append(label_encode)
		all_imgs.append(image)

	all_labels = np.concatenate(all_labels)	
	all_imgs = np.concatenate(all_imgs)	
						
	return all_imgs, all_labels, LABELS, CLASS

def CNNModel(CLASS, hidden_size=32, learning_rate=1e-3):
	model_input = Input(shape=(84, 84, 1))
	
	# TODO: network maybe too into global context for this, make it more compact
	x = Conv2D(filters=32, kernel_size=8, strides=4, padding="valid", activation="relu")(model_input)
	x = Conv2D(filters=64, kernel_size=4, strides=2, padding="valid", activation="relu")(x)
	x = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", activation="relu")(x)
	x = Flatten()(x)
	
	#outputs = []
	#for i in range(CLASS):
	#	tmp = Dense(hidden_size, activation='linear')(x)
	#	outputs.append(Dense(1, activation='softmax')(tmp))

	x = Dense(256, activation='linear')(x)
	# linear will give better results because sigmoid doesn't update as much
	outputs = Dense(CLASS, activation='sigmoid')(x)
	#outputs = Concatenate(outputs)	

	model = Model(input=model_input, output=outputs)

	#optimizerRMSprop = RMSprop(lr=self.learning_rate, rho=0.95)
	optimizerAdam = Adam(lr=learning_rate)
	optimizer = optimizerAdam

	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	#model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

	model.summary()

	return model

def get_callbacks(filepath, patience=5):
	#es = EarlyStopping('val_loss', patience=10, mode="min")
	es = EarlyStopping('val_loss', patience=20, mode="min")
	msave = ModelCheckpoint(filepath, save_best_only=True, verbose=1)
	reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
								   patience=patience, verbose=1, epsilon=1e-4, mode='min', min_lr=1e-5)
	return [es, msave, reduce_lr_loss]	

if __name__ == '__main__':
	batch_size = 8

	all_imgs, all_labels, LABELS, CLASS = parse_annotation('lev1_labeled/imgLevel1Label', 'lev1_labeled/imgLevel1Label', 'lev1_labeled/0_allpossible.txt')


	model = CNNModel(CLASS, learning_rate=1e-4)

	gen = ImageDataGenerator(horizontal_flip = False,
						 vertical_flip = False,
						 width_shift_range = 0.,
						 height_shift_range = 0.,
						 channel_shift_range=0,
						 zoom_range = 0,
						 rotation_range = 0)

	X_train, X_valid, y_train, y_valid = train_test_split(
			all_imgs, all_labels, random_state=6060, train_size=0.75)

	train_gen = gen.flow(X_train, y_train, batch_size=batch_size, seed=55)
	valid_gen = gen.flow(X_valid, y_valid, batch_size=batch_size, seed=55)

	train_step_cnt = len(X_train) / batch_size
	valid_step_cnt = len(X_valid) / batch_size

	# pretrained
	#filepath = 'CNN_state_parser_linearout_tr_0.0013_vl_0.0041.hdf5'
	filepath = 'CNN_state_parser_sigmoidBCELoss_tr_0.0182_vl_0.0135_acc_0.996.hdf5'
	#filepath = 'CNN_state_parser.hdf5'
	if os.path.exists(filepath):
		print("Loading pre-trained weights in{}".format(filepath))
		model.load_weights(filepath=filepath)
		plot_model(model, to_file='model.png')
		#filepath = 'CNN_state_parser.hdf5'
		#callbacks = get_callbacks(filepath, patience=5)
		#model.fit_generator(
		#		train_gen,
		#		steps_per_epoch=train_step_cnt,
		#		epochs=70,
		#		shuffle=True,
		#		verbose=1,
		#		validation_data=valid_gen,
		#		validation_steps=valid_step_cnt,
		#		callbacks=callbacks)
	else:
		callbacks = get_callbacks(filepath, patience=5)
		model.fit_generator(
				train_gen,
				steps_per_epoch=train_step_cnt,
				epochs=70,
				shuffle=True,
				verbose=1,
				validation_data=valid_gen,
				validation_steps=valid_step_cnt,
				callbacks=callbacks)

	res = model.predict(X_valid)
	cond1 = res < 0.5
	cond2 = res >= 0.5
	res[cond1] = 0
	res[cond2] = 1
	shape = res.shape
	print('Valid accuracy:', np.sum(res == y_valid) / shape[0] / shape[1])
	









