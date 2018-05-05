#!/usr/bin/env python
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop, Adam
import numpy as np
import random
from collections import deque
import gym
import pickle
import os, sys, copy, argparse
import cv2
from keras import backend as K
from keras.layers import Lambda
from keras.models import Model

import pdb

# reuse some of my code from drl hw, including DQN and Agent



class RLAgents():
	"""
	Different reinforcement learning for different mini-tasks
	dynamically adding new agents
	"""
	def __init__(self, state_dim, action_cnt, env_name='MZ'):
		"""
		state_dim: dimension of states, tuple: (84, 84, 4)
		action_cnt: dimension of actions, scala
		env_name: environment name, for keeping track
		"""
		self.state_dim = state_dim
		self.action_cnt = action_cnt
		self.env_name = env_name
		self.Agent_Dict = {}
	
	def _get_agent(self, mini_task):
		"""
		mini_task: string for mini_task, each agent correspond to a mini_task
		"""
		if mini_task not in self.Agent_Dict:
			self.Agent_Dict[mini_task] = DQN_Agent(self.state_dim, self.action_cnt, self.env_name, mini_task, 
												   epsilon_start=1., epsilon_end=0.1, 
												   epsilon_linear_red=0.0001, replay=True, 
												   gamma=0.99, train=True)
		return self.Agent_Dict[mini_task]

	def feedback(self, mini_task, feedbacks):
		"""
		feedback for just one state / 4 frames, used to update DQN
		feedbacks: a tuple. 
					example: 
					(state_images, action, next_states, rewards, done)
					state_images and next_states should be of shape (1, 84, 84, 4)
					action should be scala, the same action used for the 4 frames
					rewards should be a scala, the sum of rewards for the 4 frames
					done should be a boolean
		"""
		agent = self._get_agent(mini_task)
		agent.step(feedbacks)
	
	def execute(self, mini_task, states):
		"""
		choose action given mini_task and states
		mini_task: string for mini_task, each agent correspond to a mini_task
		states: 4 frames of images, should be numpy array of shape (1, 84, 84, 4)
		"""
		agent = self._get_agent(mini_task)
		
		q_values = agent.predict(states)
		action = agent.epsilon_greedy_policy(q_values)
		
		return action



class QNetwork():
	"""
	The Q-network class, which defines the Q-network architecture and essential 
	utilities, where the network take in state of the world as an input, and 
	output Q values of the actions available to the agent as the output. Note 
	that this implementation only considers finite-dimension continuous state 
	space and finite discrete action space.
	"""

	def __init__(self, state_dim, action_cnt, env_name, suffix, batch_size=32, learning_rate=1e-4, debug=False):
		"""
		Initialize a Q-network instance.

		@param state_dim The dimension of state space. (84, 84, 4)
		@param action_cnt The dimension of action space. scala
		@param env_name The name of the environment.
		@param suffix The test case suffix for the network.
		@param batch_size The size of the mini-batch in mini-batch gradient descent 
											optimization for training the network. (default: 32)
		@param learning_rate The learning rate for training the network. 
												 (default: 1e-4)
		"""

		self.state_dim = state_dim
		self.action_cnt = action_cnt

		self.env_name = env_name
		self.suffix = suffix

		self.batch_size = batch_size
		self.learning_rate = learning_rate

		self.debug = debug

		# construct default file paths
		self.default_model_path = 'save_' + self.env_name + '_' + self.suffix + '_model.h5'
		self.default_weights_path = 'save_' + self.env_name + '_' + self.suffix + '_weights.h5'

		# build the keras model
		self.model = self._build_model()

	def _build_model(self):
		"""
		(Internal)
		Build the keras model of the Q-network.

		@return A built keras model.
		"""

		model_input = Input(shape=self.state_dim)

		x = Conv2D(filters=32, kernel_size=8, strides=4, padding="valid", activation="relu")(model_input)
		x = Conv2D(filters=64, kernel_size=4, strides=2, padding="valid", activation="relu")(x)
		x = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", activation="relu")(x)
		x = Flatten()(x)
		advantage_net = Dense(activation='relu', units=512)(x)
		advantage_net = Dense(activation='linear', units=self.action_cnt)(advantage_net)
		value_net = Dense(activation='relu', units=512)(x)
		value_net = Dense(activation='linear', units=1)(value_net)

		sum_ = Lambda(lambda xin: K.sum(xin, axis=1, keepdims=True))(advantage_net)
		tile_ = Lambda(lambda xin: K.tile(1 / self.action_cnt * sum_, [1, self.action_cnt]))(sum_)

		subtracted = Subtract()([advantage_net, tile_])
		output = Add()([value_net, subtracted])

		model = Model(input=model_input, output=output)

		optimizerRMSprop = RMSprop(lr=self.learning_rate)
		optimizerAdam = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999)
		optimizer = optimizerAdam # TODO

		model.compile(loss='mse', optimizer=optimizer)

		return model

	def train(self, X, Y, epochs=1, verbose=0):
		"""
		Train the Q-network with batch data.

		@param X The batch input training data.
		@param Y The batch output training data.
		@param epochs The number of training epochs for this batch of training data. 
								 (default: 1)
		@param verbose The switch of printing out verbose information. (default: 0)
		"""

		self.model.fit(X, Y, batch_size=self.batch_size, epochs=epochs, verbose=verbose)

	def predict(self, state):
		"""
		Predict the Q-values of all actions at a state.

		@param state The state at which to predict the Q-values of all actions.
		@return The Q-values of all actions at the state.
		"""

		prediction = self.model.predict(state)
		return prediction

	def save_model(self, file_path=None):
		"""
		Save the keras model as a file.

		@param file_path The file path to save the keras model. (optional)
		@return The path to the saved keras model file.
		"""
		
		if file_path is None:
			file_path = self.default_model_path

		if self.debug:
			print("-----saving model to {}".format(file_path))
			
		self.model.save(file_path)

		return file_path

	def save_weights(self, file_path=None):
		"""
		Save the keras model weights as a file.

		@param file_path The file path to save the keras model weights. (optional)
		@return The path to the saved keras weights file.
		"""

		if file_path is None:
			file_path = self.default_weights_path

		if self.debug:
			print("-----saving weights to {}".format(file_path))
			
		self.model.save_weights(file_path)

		return file_path

	def load_model(self, file_path=None):
		"""
		Load an existing keras model from a file.

		@param file_path The path to an existing keras model file. (optional)
		"""

		if file_path is not None:
			if self.debug:
				print("-----loading model from {}".format(file_path))
			self.model.load(file_path)
		else:
			if self.debug:
				print("-----loading model from {}".format(self.default_model_path))
			self.model.load(self.default_model_path)

	def load_weights(self, file_path=None):
		"""
		Load existing keras model weights from a file.

		@param file_path The path to an existing keras weights file. (optional)
		"""

		if file_path is not None:
			if self.debug:
				print("-----loading weights from {}".format(file_path))
			self.model.load_weights(file_path)
		else:
			if self.debug:
				print("-----loading weights from {}".format(self.default_weights_path))
			self.model.load_weights(self.default_weights_path)





class Replay_Memory():
	"""
	The replay memory, which stores the experience the agent visited and provide 
	experience retrival utility.
	"""

	def __init__(self, env_name, suffix, memory_size=1000000, burn_in=10000, debug=False):
		"""
		Initialize a replay memory.

		@param env_name The name of the environment.
		@param suffix The test case suffix for the network.
		@param memory_size The size of the memory. (default: 50000)
		@param burn_in The required burn-in memory size before replay. 
									 (default: 10000)
		"""

		self.env_name = env_name
		self.suffix = suffix
		
		self.debug = debug

		self.memory_size = memory_size
		self.burn_in = burn_in

		self.default_memory_path = 'save_' + self.env_name + '_' + self.suffix + '_memory.pkl'

		self.memory = deque(maxlen=self.memory_size)

	def sample_batch(self, batch_size=32):
		"""
		Randomly sample a batch of transitions from the memory.

		@param batch_size The size of the transition batch. (default: 32)
		@return The sampled batch of transitions as an array. Note that if the size 
						of the memory does not meet the burn-in size required, a `None` 
						will be returned.
		"""

		if len(self.memory) < self.burn_in:
			return None

		batch_size = min(batch_size, len(self.memory))
		samples = random.sample(self.memory, batch_size)
		return samples

	def append(self, transition):
		"""
		Append a transition to the memory.
		"""

		self.memory.append(transition)

	def save(self, file_path=None):
		"""
		Save the replay memory as a file.

		@param file_path The file path to save the keras model. (optional)
		@return The path to the saved keras model file.
		"""

		if file_path is None:
			file_path = self.default_memory_path

		if self.debug:
			print("-----saving memory to {}".format(file_path))
				
		with open(file_path, 'wb') as replay_memory_file:
			pickle.dump(self.memory, replay_memory_file)

		return file_path

	def load(self, file_path=None):
		"""
		Load existing replay memory from a file.

		@param file_path The path to an existing replay memory file. (optional)
		"""

		if file_path is None:
			file_path = self.default_memory_path

		if self.debug:
			print("-----loading memory from {}".format(file_path))
			
		with open(file_path, 'rb') as replay_memory_file:
			loaded_memory = pickle.load(replay_memory_file)
			self.memory = deque(maxlen=self.memory_size)
			self.memory.extend(loaded_memory)



class DQN_Agent():
	"""
	The DQN agent.
	"""

	def __init__(self, state_dim, action_cnt, env_name, mini_task, epsilon_start=1., epsilon_end=0.1, 
				 epsilon_linear_red=0.0001, replay=False, gamma=0.99, train=True, save_every_step=1000, debug=False):
		"""
		Initialize a DQN agent.

		@param state_dim The dimension of states. (84, 84, 4)
		@param action_cnt The number of actions. scala
		@param env_name The name of the environment.
		@param mini_task The mini task for the network.
		@param replay The switch to turn on memory replay. (default: False)
		@param gamma The discount factor. (default: 0.99)
		@param train The switch to train the model. (default: True)
		"""

		self.state_dim = state_dim
		self.action_cnt = action_cnt
		# initialize the full actions array
		self.all_actions = [i for i in range(self.action_cnt)]
		
		self.env_name = env_name
		self.mini_task = mini_task

		self.replay = replay
		self.gamma = gamma
		self.debug = debug

		self.should_train = train
		
		self.epsilon = epsilon_start
		self.epsilon_end = epsilon_end
		self.epsilon_linear_red = epsilon_linear_red

		# initialize the replay memory
		self.replay_memory = None
		if self.replay:
			self.memory_size = 3000
			self.burn_in = 1000
			self.replay_memory = Replay_Memory(self.env_name, self.mini_task, self.memory_size, self.burn_in)

		# initialize the Q-network and load weights if exist
		self.batch_size = 32
		self.learning_rate = 0.00025
		self.network = QNetwork(self.state_dim, self.action_cnt, 
														self.env_name, self.mini_task,
														self.batch_size, self.learning_rate)
		self.step_cnt = 0
		self.save_every_step = save_every_step
		
		# load the saved model weights
		if os.path.isfile(self.network.default_weights_path):
			self.network.load_weights()

		# load the replay memory
		if train and replay and os.path.isfile(self.replay_memory.default_memory_path):
			self.replay_memory.load()


	def predict(self, s):
		"""
		Predict the Q-values of all actions at a state.

		@param s The state of at which the agent predicts Q-values of all actions.
		@return The Q-values of all actions at the state.
		"""

		#s = np.reshape(s, (1, self.state_space.shape[0]))

		return self.network.predict(s)

	def epsilon_greedy_policy(self, q_values):
		"""
		Choose an action to take greedily with epsilon probability to do random 
		exploration.

		@param q_values The q-values of all actions.
		@param epsilon The randomness factor.
		@return The action chosen by the policy.
		"""

		action = np.argmax(q_values[0])

		if np.random.random() < self.epsilon:
			action = np.random.choice(self.all_actions)

		return action

	def greedy_policy(self, q_values):
		"""
		Choose an action to take greedily.

		@param q_values The q-values of all actions.
		@return The action chosen by the policy.
		"""

		action = np.argmax(q_values[0])

		return action

	def step(self, feedbacks):
		"""
		Step forward to the next state.

		feedbacks: a tuple. 
					example: 
						(state_images, actions, next_states, rewards, done)
						state_images and next_states should be of shape (1, 84, 84, 4)
						action should be a scala
						rewards should be a scala, the sum of rewards for the 4 frames
						done should be a boolean
		"""
		
		# update epsilon
		self.epsilon -= self.epsilon_linear_red
		self.epsilon = max(self.epsilon, self.epsilon_end)

		#state_images, action, state_nexts, rewards, real_done = parse_feedbacks(feedbacks)
		state_images, action, state_nexts, rewards, real_done = feedbacks

		# check training flag
		if not self.should_train:
			return

		samples = None
		# remember new transition and sample from the replay memory
		if self.replay:
			# append trainsition to the memory
			self.replay_memory.append((state_images, action, state_nexts, rewards, real_done))

			# sample from the memory
			samples = self.replay_memory.sample_batch(self.batch_size)

		# sample from the new experience
		if samples is None:
			samples = [(state_images, action, state_nexts, rewards, real_done)]

		# predict using the prior weights
		X = np.zeros([len(samples)] + list(self.state_dim)) # (len(samples), 84, 84, 4)
		X_next = np.zeros([len(samples)] + list(self.state_dim))
		for i_sample in range(len(samples)):
			(state_images, action, state_nexts, rewards, real_done) = samples[i_sample]
			X[i_sample] = state_images
			X_next[i_sample] = state_nexts

		q_s_set = self.network.predict(X)
		q_s_next_set = self.network.predict(X_next)

		# update the Q-values
		Y = np.zeros((len(samples), self.action_cnt))
		for i_sample in range(len(samples)):
			state_images, action, state_nexts, rewards, done = samples[i_sample]

			Y[i_sample] = np.array(q_s_set[i_sample])
			next_action = np.argmax(q_s_next_set[i_sample]) # TODO: fixed a bug, action -> next_action
			if done:
				Y[i_sample, action] = rewards
			else:
				Y[i_sample, action] = rewards + self.gamma * q_s_next_set[i_sample][next_action] # changed to dueling

		# train the network
		self.network.train(X, Y)
		
		# save DQN and memory
		self.step_cnt += 1
		if self.should_train and self.step_cnt % self.save_every_step == 0 and self.step_cnt != 0:
			# save the latest model weights
			self.network.save_weights()

			# save the replay memory
			if self.replay:
				self.replay_memory.save()


