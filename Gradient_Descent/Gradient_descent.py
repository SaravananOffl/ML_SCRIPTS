import pandas as pd
from numpy import *
import pandas
# Gradient Descent
def compute_error_for_gn_points(b, m, points):
	totalError = 0
	for i in range(points):
		x = points[i,0]
		y = points[i, 1]

		# error = (1/N) sum((yi - (mxi +b))^2)

		totalError += (y -(m*x +b))**2
	return totalError/float(len(points))

def step_gradient(current_b, current_m, points, learning_rate):
	#gradient descent implementation
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))

	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]

		b_gradient += -(2/N) * (y - ((current_m*x)+ current_b))
		m_gradient += -(2/N) * x * (y - ((current_m*x)+ current_b))

	new_b = current_b - (learning_rate *b_gradient)
	new_m = current_m  - (learning_rate * m_gradient)

	return [new_b, new_m]

def gradient_descent_runner(points,initial_m,initial_b,learning_rate, num_iterations):
	b= initial_b
	m= initial_m

	for i in range(num_iterations):
		b,m = step_gradient(b, m, array(points), learning_rate)
	return [b,m]

def run():
	points = pandas.read_csv('data.csv.txt')

	#hyperparameters
	learning_rate = 0.0001
	
	#y = mx + b

	initial_m = 0
	initial_b = 0
	num_iterations = 1000

	[b,m] = gradient_descent_runner(points,initial_m,initial_b,learning_rate,num_iterations)

	print("Equation is : "+ str(m) + "x" + "+" +str(b))





if __name__ == '__main__':
	run()