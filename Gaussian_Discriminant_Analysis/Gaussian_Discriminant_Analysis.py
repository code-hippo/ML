#!usr/env/bin python

import time
import numpy as np
import math
from math import exp
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sympy.plotting import plot_implicit
from sympy import symbols, Eq
from sympy.parsing.sympy_parser import parse_expr

file1   =   open("x.dat","r")
file2   =   open("y.dat","r")

X0 = []
Y = []
X0.append([])
X0.append([])
for line in file1 :
    line = line.split()
    X0[0].append(float(line[0]))
    X0[1].append(float(line[1]))

Xlen = len(X0[0])
X= np.empty(shape = [0,2])
for i in range(Xlen) :
    X =  np.append(X,[[X0[0][i],X0[1][i]]],axis=0)

for line in file2 :
    if(line=="Alaska\n") :
        Y.append([float(0)])
    else : Y.append([float(1)])
Y = np.array(Y)

def GDA():
    negY = np.array([1-i for i in Y])
    phi = (sum(i[0] for i in Y))/len(Y)
    myu0 = np.dot(np.transpose(negY),X)/((1-phi)*len(negY))
    myu1 = np.dot(np.transpose(Y),X)/(phi*len(Y))
    Sig = (1.0/len(Y))*np.dot(np.transpose(X-[myu0[0]]*len(Y)-np.dot(Y,myu1-myu0)),((X-[myu0[0]]*len(Y)-np.dot(Y,myu1-myu0))))
    print myu0, "myu0"
    print myu1, "myu1"
    print Sig,"sig"
    Sig1 = (1.0/(phi*len(Y)))*np.dot(np.transpose(Y)*np.transpose(X-[myu0[0]]*len(Y)-np.dot(Y,myu1-myu0)),((X-[myu0[0]]*len(Y)-np.dot(Y,myu1-myu0))))
    Sig0 = (1.0/((1-phi)*len(Y)))*np.dot(np.transpose(negY)*np.transpose(X-[myu0[0]]*len(Y)-np.dot(Y,myu1-myu0)),((X-[myu0[0]]*len(Y)-np.dot(Y,myu1-myu0))))

    print Sig0,"Sig0"
    print Sig1, "Sig1"
    plotdata()
    find0(phi,myu0,myu1,Sig)
    # findQ0(phi,myu0,myu1,Sig,Sig0,Sig1)

def plotdata():

    # ax = fig.add_subplot(111,projection="3d")
    fig = plt.figure()
    g = [[],[],[],[],[],[]]

    for i in range(len(Y)):
        if(Y[i][0]==0):
            g[0].append((X[i][0]))
            g[1].append((X[i][1]))
            g[2].append(1)
        else:
            g[3].append((X[i][0]))
            g[4].append((X[i][1]))
            g[5].append(1)
    g = np.array(g)
    plt.plot(g[0], g[1], 'ro' )
    plt.plot(g[3], g[4],'b^' )




def find0(phi,myu0,myu1,sig):

    print sig,"sig2"
    print myu0, "myu0 "
    print myu1 , "myu1"
    print inv(sig),"sig"
    Theta0 = .5*(np.dot(np.dot((myu0),inv(sig)),np.transpose(myu0))-
                np.dot(np.dot((myu1),inv(sig)),np.transpose(myu1)))- math.log((1-phi)/phi)
    Theta1 = np.dot(inv(sig),np.transpose(myu1))-np.dot(inv(sig),np.transpose(myu0))
    Theta = np.array([Theta0[0],Theta1[0],Theta1[1]])
    h=[[],[]]
    x1 = np.arange(60,200,1)
    x2 = np.arange(60,200,1)
    h[0] = x1
    h[1] = (-1*np.dot([Theta[0][0],Theta[1][0]],[[1]*len(x1),x1]))/Theta[2][0]
    plt.plot(h[0],h[1])
    plt.show()

# def ezplot(s):                                # Can BE USed Later
    #Parse doesn't parse = sign so split
    # lhs, rhs = s.replace("^","**").split("=")
    # eqn_lhs = parse_expr(lhs)
    # eqn_rhs = parse_expr(rhs)
    # plot_implicit(eqn_lhs-eqn_rhs)

def findQ0(phi,myu0,myu1,sig,sig0,sig1):
    x1 , x2 = symbols('x1 x2')
    # p2 = plot_implicit(Eq(x1**2+x2**2,2))
    p1 =plot_implicit( Eq(-0.5*(([[x1, x2]]-myu0)*inv(sig)*(np.transpose([[x1 ,x2]]-myu0)))[0][0]+0.5*(([[x1, x2]]-myu1)*inv(sig)*(np.transpose([[x1 ,x2]]-myu1)))[0][0] + math.log(float(1-phi)/phi) ,0),(x1, 60, 200), (x2, 60, 200))

    ezplot('-0.5*((np.transpose([[x1],[x2]]-mu0))*inv(cov_mat1)*([[x1],[x2]]-mu0)- (np.transpose([[x1],[x2]]-mu1))*inv(cov_mat1)*([[x1],[x2]]-mu1) + math.log(float(1-phi)/phi)) = [[0]]')

if __name__ == "__main__":
    GDA()
