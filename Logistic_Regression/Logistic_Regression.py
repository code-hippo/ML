#!usr/bin/env python
import numpy as np
from math import floor
import time
from numpy.linalg import inv
from math import exp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


file1=open("x.dat","r")
file2=open("y.dat","r")

Area0 = []
Area0.append([])
Area0.append([])
prices = []
for line in file1:
    line=line.split()
    Area0[0].append(float(line[0]))
    Area0[1].append(float(line[1]))
Area = np.empty(shape=[0,3])
Arealen = len(Area0[0])
for i in range(Arealen):
    Area = np.append(Area,[[1.0,float(Area0[0][i]),float(Area0[1][i])]],axis=0)
for line in file2:
    prices.append([float(line)])
prices = np.array(prices)


def BGD():
    Theta = np.array([[0,0,0]])
    while(True):
        x0T = np.dot(Area,np.transpose(Theta))
        h0x = np.array([[1.0/(1.0+exp(-1.0*i))] for i in x0T])
        hneg0x = np.array([[1-(1.0/(1.0+exp(-1.0*i)))] for i in x0T])
        dL0 = np.dot(np.transpose(prices-h0x),Area)
        ddL0 =  (np.dot(np.transpose(-1*h0x),(hneg0x))*np.dot(np.transpose(Area),Area))
        Theta1 = Theta - np.dot(dL0,inv(ddL0))

        if(sum((Theta1[0]-Theta[0]) *(Theta1[0]-Theta[0]))  < .0000000001 ):
            Theta = Theta1
            break
        else : Theta = Theta1
    print Theta , "Theta"
    plotdata(Theta)


def plotdata(Theta):
    g = [[],[],[],[],[],[]]
    for i in range(len(prices)):
        if(prices[i][0]==0):
            g[0].append((Area[i][1]))
            g[1].append((Area[i][2]))
            g[2].append(1)
        else:
            g[3].append((Area[i][1]))
            g[4].append((Area[i][2]))
            g[5].append(1)
    g = np.array(g)
    plt.plot(g[0],g[1],'ro')
    plt.plot(g[3],g[4],'b^')

    h=[[],[]]
    x1 = np.arange(0,10,.1)
    x2 = np.arange(0,10,.1)
    h[0] = x1
    h[1] = (-1*np.dot([Theta[0][0],Theta[0][1]],[[1]*len(x1),x1]))/Theta[0][2]
    plt.plot(h[0],h[1])
    plt.show()


if __name__ =="__main__" :
    BGD()
