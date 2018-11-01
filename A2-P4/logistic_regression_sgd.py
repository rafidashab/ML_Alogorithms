#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2
import random

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
eta = 0.05


# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
X = data[:,0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:,3]
# For plotting data
class1 = np.where(t==0)
X1 = X[class1]
class2 = np.where(t==1)
X2 = X[class2]

eta = np.array([0.5, 0.3, 0.1 , 0.05, 0.01])

# Initialize w.
w = np.array([0.1, 0, 0])

# Error values over all iterations.
e_all = []

DATA_FIG = 1

# Set up the slope-intercept figure
# SI_FIG = 2
# plt.figure(SI_FIG)
# plt.rcParams.update({'font.size': 15})
# plt.title('Separator in slope-intercept space')
# plt.xlabel('slope')
# plt.ylabel('intercept')
# plt.axis([-5, 5, -10, 0])


plt.figure()

for j in range (0,5): 

  print ("Step size %.2f" %eta[j])
  # Initialize w.
  w = np.array([0.1, 0, 0])

  # Error values over all iterations.
  e_all = []

  for iter in range (0,max_iter):


    for i in range (200):

      #extract a random data point from the data set

      test = X[i,:]
      # Compute output using current w on the selected data point
      y = sps.expit(np.dot(test,w))
      
      # e is the error, negative log-likelihood (Eqn 4.90)
      

      # Add this error to the end of error vector.
      

      # Gradient of the error, using Eqn 4.91
      grad_e = np.multiply((y - t[i]), test.T)

      # Update w, *subtracting* a step in the error derivative since we're minimizing
      w_old = w
      w = w - eta[j]*grad_e

    y = sps.expit(np.dot(X,w))
    e = -np.mean( np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-y)) )
    e_all.append(e)

    # Plot current separator and data.  Useful for interactive mode / debugging.
    # plt.figure(DATA_FIG)
    # plt.clf()
    # plt.plot(X1[:,0],X1[:,1],'b.')
    # plt.plot(X2[:,0],X2[:,1],'g.')
    # a2.draw_sep(w)
    # plt.axis([-5, 15, -10, 10])

    # Add next step of separator in m-b space.
    # plt.figure(SI_FIG)
    # a2.plot_mb(w,w_old)

  
    # Print some information.
    print ('epoch {0:d}, negative log-likelihood {1:.4f}, w={2}'.format(iter, e, w.T))
      # Stop iterating if error doesn't change more than tol.
    if iter>0:
      if np.absolute(e-e_all[iter-1]) < tol:
        break

  plt.plot(e_all)
  plt.ylabel('Negative log likelihood')
  plt.title('Training logistic regression')
  plt.xlabel('Epoch')
  #plt.figtext(.70, .70-.10*i, "Step size = %.2f" %eta[i])
  plt.legend(eta)
  # plt.axis([0, 500, 0, 3])

#Shows all five figures. But how to condense them in one plot
plt.show()  


