#!usr/bin/env python
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib.mlab import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

file1=open("x.dat","r")
file2=open("y.dat","r")
Area0 = []
Prices = []
Theta = []
for line in file1:
    Area0.append(float(line))
Area0 = [float(x)/sum(Area0) for x in Area0]
Area = np.empty(shape=[0,2])
Arealen = len(Area0)

for i in Area0:
    Area = np.append(Area,[[1.0/Arealen,i]],axis=0)

for line in file2:
    Prices.append([float(line)])
Prices = np.array(Prices)


def BGD():
    Theta = np.array([[-77.5,1224.5]])
    eta = 2.5 #{0.1, 0.5, 0.9, 1.3, 2.1, 2.5}                       #learning rate
    Stopcriteria    =0.0000001          #Stopping Criteria
    jtheta0 = .5*np.dot(np.transpose(Prices-np.dot(Area,np.transpose(Theta))),Prices-np.dot(Area,np.transpose(Theta)))
    h = [[],[],[]]
    h[0].append(Theta[0][0])
    h[1].append(Theta[0][1])
    h[2].append(jtheta0[0][0])
    val =0
    while(True):
        Theta = Theta+eta*(np.transpose(np.dot(np.transpose(Area),Prices-np.dot(Area,np.transpose(Theta)))))
        jtheta1 = .5*np.dot(np.transpose(Prices-np.dot(Area,np.transpose(Theta))),Prices-np.dot(Area,np.transpose(Theta)))
        # time.sleep(1)                  # sleep fn
        val+=1

        # print Theta,jtheta1,"Theta"    # To show Convergence
        if(val%10==0):
            h[0].append(Theta[0][0])
            h[1].append(Theta[0][1])
            h[2].append(jtheta1[0][0])
        if(jtheta0-jtheta1 <Stopcriteria):   # stopping criteria
            jtheta0=jtheta1
            break
        else : jtheta0=jtheta1
    print eta, "learning rate"              # To generate params  --> 1a
    print Stopcriteria , "Stopping Criteria"
    print Theta , "Theta"
    print jtheta0,"jtheta"

    # graph(Theta)                # To generate Graph of points --> 1b
    # mesh(h)                    # To generate j0,01,02 mesh  --> 1c
    contour(h)                 # To generate j0,01,02 contoure --> 1d

def graph(Theta):
    print Theta , "T1"
    g = [[],[],[]]
    g[0] = np.transpose(Area)[0]
    g[1] = np.transpose(Area)[1]
    g[2] = np.transpose(Prices)[0]

    g = np.array(g)
    plt.plot(g[1],g[2],'ro')

    h = [[],[]]
    h[0] = [-.0022,.003,.04]
    h[1]= np.dot(Theta,[[1.0/len(Prices)]*3,[-.0022,.003,.04]])[0]
    plt.plot(h[0],h[1])
    plt.show()


def j0(Theta):
    jtheta0 = .5*np.dot(np.transpose(Prices-np.dot(Area,np.transpose(Theta))),Prices-np.dot(Area,np.transpose(Theta)))
    return jtheta0[0][0]


def mesh(h):
    fig = plt.figure()
    h= np.array(h)
    xs = np.arange(-600,0,10)
    ys = np.arange(600,1200,10)

    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(xs,ys)
    zs = np.array([j0([[x,y]]) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap=cm.jet,
                       linewidth=1, antialiased=True)
    ax.set_zlim3d(min(h[2]), max(h[2]))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def contour(h):
    fig = plt.figure()
    h= np.array(h)
    xs = np.arange(-677.5,-77.5,10)
    ys = np.arange(724.5,1224.5,10)

    X, Y = np.meshgrid(xs,ys)
    zs = np.array([j0([[x,y]]) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(Y.shape)
    print " len", "X", np.shape(zs),np.shape(X),np.shape(Y),np.shape(Z)
    print Z , "Z"
    # ax= fig.add_subplot(111,projection = '3d')     # For 3D contour
    # ax.contour(X,Y,Z,100)
    plt.contour(X, Y ,Z,60)
    plt.plot(h[0],h[1],'ro')
    plt.show()

if __name__ =="__main__" :
    BGD()
