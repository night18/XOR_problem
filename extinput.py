'''
Created on Oct 1, 2018
@author: Chun-Wei Chiang
'''
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

#input and output
x = np.array([[1,1,1],[1,0,0],[0,1,0],[0,0,1],[0,0,0],[1,0,1],[0,1,1],[1,1,0]])
y = np.array([[1],[1],[1],[1],[0],[0],[0],[0]])
learning_rate = 0.9
learning_rate_list = [1,0.1,0.01]

#initial weight. Using 8 weights because there are 7 inputs and 1 constant
w = np.random.rand(8,1)
foreverW = np.array(w, copy=True)  

def inputLayer(data): #handle the input layer
	x1 = data[0]
	x2 = data[1]
	x3 = data[2]
	#the last one unit is the constant
	return np.array([x1, x2, x3, x1*x2, x1*x3, x2*x3, x1*x2*x3, 1])

def stepUnit(x):
	if x >= 0.5:
		return 1
	else:
		return 0

def train(data, target):
	global w

	output = stepUnit(data.dot(w))

	delta = target - output
	#update weights
	for i in xrange(8):
		w[i] = w[i] + learning_rate * delta * data[i]

def validate():
	o = np.full((8,1),0.0)

	for idx, data in enumerate(x):
		a =  inputLayer(data)
		o[idx] = stepUnit(a.dot(w))

	#Root Mean Square Error
	error = np.power(y - o, 2)
	return o, (np.sum(error)/len(error))**(0.5)

iteration = np.arange(0, 200, 1)
draw = np.full((3,200),0.0)

for ln in xrange(3):
	w = np.array(foreverW, copy=True)  
	pprint(w)

	learning_rate = learning_rate_list[ln]
	for i in xrange(200):
		for idx, data in enumerate(x):
			a = inputLayer(data)
			train(a, y[idx])

		output, rms = validate()
		draw[ln][i] = rms

		# if (i+1) %50 == 0:
		# 	pprint("iteration: " + str(i))
		# 	pprint("loss: " + str(rms))
		# 	for idx, data in enumerate(x):
		# 		pprint("x: {} target: {} output: {}".format(data,y[idx],output[idx]) )
	pprint(w)

plt.plot(iteration,draw[0],'r', iteration,draw[1],'b', iteration,draw[2],'g')
plt.legend(('1', '0.1', '0.01'),loc='upper right',title="learning rate")
plt.xlabel("iteration times")
plt.ylabel("RMS")
plt.grid(True)
plt.show()


