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

#initial weight. Using 3*4 weights because there are 3*3 inputs and 3*1 constant
w1 = 2 * np.random.rand(4,3) -1
w2 = 2 * np.random.rand(4,1) -1
foreverW1 = np.array(w1, copy=True) 
foreverW2 = np.array(w2, copy=True) 


def sigmoid(z):
    return 1/(1+np.exp(-z))

def inputLayer(data): #handle the input layer
	x1 = data[0]
	x2 = data[1]
	x3 = data[2]
	#the last one unit is the constant
	return np.array([x1, x2, x3, 1])

def stepUnit(x):
	if x >= 0.5:
		return 1
	else:
		return 0

def train(data, target):
	global w1 , w2
	h_in = data.dot(w1)
	h_pro = sigmoid(h_in)
	h_out = inputLayer(h_pro) #hidden layer is nonlinearity and continuous, therefore, use sigmoid 
	output = stepUnit(h_out.dot(w2))
	# pprint(output)

	delta2 = target - output  #constant
	for i in xrange(4):
		#update weights 2
		w2[i] = w2[i] + learning_rate * h_out[i] * delta2 #I use delta because there are no sigmoid at the output layer

	error1 = delta2.dot(w2.T)
	error1 = error1[:-1] #remove the constant
	delta1 = error1 * (1 - h_pro) * h_pro
	delta1.shape = 	(3,1)
	data.shape = (4,1)
	w1 = w1 + learning_rate * data.dot(delta1.T)
	
def validate():
	o = np.full((8,1),0.0)

	for idx, data in enumerate(x):
		a =  inputLayer(data)
		h_in = a.dot(w1)
		h_out = inputLayer(sigmoid(h_in)) #hidden layer is nonlinearity and continuous, therefore, use sigmoid 
		o[idx] = stepUnit(h_out.dot(w2))

	#Root Mean Square Error
	error = np.power(y - o, 2)
	return o, (np.sum(error)/len(error))**(0.5)


iteration = np.arange(0, 1000, 1)
draw = np.full((3,1000),0.0)

# for idx, data in enumerate(x):
# 	train(inputLayer(data), y[idx])
# output, rms = validate()

for ln in xrange(3):
	w1 = np.array(foreverW1, copy=True)  
	w2 = np.array(foreverW2, copy=True)  
	learning_rate = learning_rate_list[ln]

	for i in xrange(1000):
		for idx, data in enumerate(x):
			train(inputLayer(data), y[idx])
		output, rms = validate()
		draw[ln][i] = rms

		# if (i+1) % 200 == 0:
		# 	pprint("iteration: " + str(i))
		# 	pprint("loss: " + str(rms))
		# 	for idx, data in enumerate(x):
		# 		pprint("x: {} target: {} output: {}".format(data,y[idx],output[idx]) )

plt.plot(iteration,draw[0],'r', iteration,draw[1],'b', iteration,draw[2],'g')
plt.legend(('1', '0.1', '0.01'),loc='upper right',title="learning rate")
plt.xlabel("iteration times")
plt.ylabel("RMS")
plt.grid(True)
plt.show()