#!usr/bin/env python

from numpy.linalg import inv
import numpy as np
import time
from math import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.mlab import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

file1=open("x.dat","r")
file2=open("y.dat","r")
Area0 = []
Prices = []
for line in file1:
    Area0.append(float(line))
Area = np.empty(shape=[0,2])
Arealen = len(Area0)

for i in Area0:
    Area = np.append(Area,[[1.0,i]],axis=0)
for line in file2:
    Prices.append([float(line)])
Prices = np.array(Prices)

def BGD():
    Theta = np.array([[0,0]])
    Theta = np.dot(inv(np.dot(np.transpose(Area),Area)),np.dot(np.transpose(Area),Prices))

    print Theta,"Theta"                  # Calucluated from Analytical Solution
    # graph(Theta)                         # graph from Analytical sol

    X = [[6,7]]
    for x in X:                             # Reduntant loop from weightedgraph
        w = []
        eta = 1
        Arealen = len(Area)
        for xi in range(Arealen):
             p = [0]*len(Area)
             p[xi] = 10**(float(sum((x-Area[xi])*(x-Area[xi])))/(2*eta**2))
             w.append(p)
    Theta = np.dot(inv(np.dot(np.transpose(Area),np.dot(w,Area))),np.dot(np.transpose(Area),np.dot(w,Prices)))
    # print Theta
    weightedgraph()                      # Locally weighted Graph

def graph(Theta):
    g = [[],[],[]]
    g[0] = np.transpose(Area)[0]
    g[1] = np.transpose(Area)[1]
    g[2] = np.transpose(Prices)[0]
    g = np.array(g)
    plt.plot(g[1],g[2],'ro')

    h = [[],[]]
    h[0] = np.transpose(Area)[1]
    h[1]= np.dot(np.transpose(Theta),[[1.0/len(Prices)]*len(Prices),np.transpose(Area)[1]])[0]
    plt.plot(h[0],h[1])
    plt.show()

def weightedgraph() :

    g = [[],[],[]]
    g[0] = np.transpose(Area)[0]
    g[1] = np.transpose(Area)[1]
    g[2] = np.transpose(Prices)[0]
    g = np.array(g)
    plt.plot(g[1],g[2],'ro')


    Area0 = sorted(Area, key=lambda a_entry: a_entry[1])
    etaVals = [.1,.3,2,10]
    print etaVals , "eta"
    for eta in etaVals:
        l = [ [],[]]
        for x in Area0:
            w = []
            Arealen = len(Area)

            for xi in range(Arealen):
                 p = [0]*len(Area)
                 p[xi] = exp(-1*float(sum((x-Area[xi])*(x-Area[xi])))/(2*(eta**2)))
                 w.append(p)

            Theta = np.dot(inv(np.dot(np.transpose(Area),np.dot(w,Area))),np.dot(np.transpose(Area),np.dot(w,Prices)))
            l[0].append(x[1])
            l[1].append(np.dot(np.transpose(Theta),np.transpose([x]))[0][0])
        plt.plot(l[0],l[1])
        # plt.title()
    plt.show()

if __name__ =="__main__" :
    BGD()
