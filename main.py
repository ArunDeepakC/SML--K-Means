# Strategy-1- randomly pick the initial centers from the given samples.
import scipy.io
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as py
import random
Numpyfile= scipy.io.loadmat('AllSamples.mat')
tr_x=Numpyfile['AllSamples']
#Defining function to calculate distance 
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
random.seed(0)
np.random.seed(0)
initial=0
#Loop to do the entire clustering twice with different intialization
while initial<=1:
    J=list()
#Loop to cluster using different k values
    for k in range(2,11):
        C=list()
        #Generate random cluster centroids
        for i in range(k):
            
            val=random.randint(0,299)
            C.append(tr_x[val])
        C=np.array(C)
        count=0
        su=0
        Cprev = np.zeros(C.shape)
        clusters = np.zeros(len(tr_x))
        # Stop criteria Distance between new centroids and previous centroids
        stop= dist(C, Cprev, None)
        # Loop will run till the centroids change no more
        while stop != 0:
            # Assigning each value to its closest cluster
            for i in range(len(tr_x)):
                distances = dist(tr_x[i], C)
                cluster = np.argmin(distances)
                clusters[i] = cluster
            # Storing the previous centroid values
            Cprev = deepcopy(C)
            # Finding the new centroids by taking the average value
            for i in range(k):
                points = [tr_x[j] for j in range(len(tr_x)) if clusters[j] == i]
                C[i] = np.mean(points, axis=0)
               
                    
            stop= dist(C, Cprev, None)
        pt=list()
        ct=list()
        #Calculating Objective function J
        for i in range(k):
            for j in range(len(tr_x)):
                if clusters[j] == i:
                    pt.append(tr_x[j])
                    ct.append(C[i])
        su=np.sum(dist(np.array(pt),np.array(ct)))
        J.append(su)
        
    
    #Plotting J(k) vs k graph    
    K=[k for k in range(2,11)]
    py.title("K-Means using Strategy-1")
    py.ylabel('Objective Function')
    py.xlabel('number of clusters k')
    l='Initialization-'+str(initial+1)
    py.plot(K,J,label=l,marker='o')
    py.legend()
    print("Objective function values Initialization:"+str(initial+1))
    print("K:"+str(K))
    print("J(K):"+str(J))
    print("\n")
    
    initial+=1
py.show()

###############################################################################################################################################################################################################
#Strategy-2- Pick Centroid with maximum distance
import scipy.io
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as py
import random
import math
Numpyfile= scipy.io.loadmat('AllSamples.mat')
tr_x=Numpyfile['AllSamples']
z=0

#Defining function to calculate distance 
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
def dist_pt(pt1,pt2):
    x1,y1=pt1
    x2,y2=pt2
    x=x1-x2
    y=y1-y2
    return math.sqrt(x*x+y*y)

initial=0
x='c'
#Loop to do the entire clustering twice with different intialization
while initial<=1:
    #initialize first centroid randomly
    random.seed(z)
    J=list()
    val=random.randint(0,299)
    C=np.array([tr_x[val],])

    for k in range(2,11):
        #choosing i th centroid, a sample such that the average distance of this chosen one to all previous (i-1) centers is maximal.
        mean_dist=list()
        for i in range(len(tr_x)):
            d_sum=0
            for j in range(len(C)):
                d_sum=d_sum+dist_pt(C[j],tr_x[i])
            mean_dist.append(d_sum/len(C))
        mean_dist=np.array(mean_dist)
        val=np.argmax(mean_dist)
        C=np.vstack([C,tr_x[val]])

        su=0

        Cprev = np.zeros(C.shape)
        clusters = np.zeros(len(tr_x))
          # Stop criteria Distance between new centroids and previous centroids
        stop = dist(C, Cprev, None)
       # Loop will run till the centroids change no more
        while stop != 0:
            # Assigning each value to its closest cluster
            for i in range(len(tr_x)):
                distances = dist(tr_x[i], C)
                cluster = np.argmin(distances)
                clusters[i] = cluster
            # Storing the previous centroid values
            Cprev = deepcopy(C)
            # Finding the new centroids by taking the average value
            for i in range(k):
                points = [tr_x[j] for j in range(len(tr_x)) if clusters[j] == i]
                C[i] = np.mean(points, axis=0)
                
            stop = dist(C, Cprev, None)
        pt=list()
        ct=list()
         #Calculating Objective function J
        for i in range(k):
            for j in range(len(tr_x)):
                if clusters[j] == i:
                    pt.append(tr_x[j])
                    ct.append(C[i])
        su=np.sum(dist(np.array(pt),np.array(ct)))
        J.append(su)
   #Plotting J(k) vs k graph        
    K=[l for l in range(2,11)]
    py.title("K-Means Strategy-2")
    l='Initialization-'+str(initial+1)
    py.ylabel('Objective Function')
    py.xlabel('number of clusters k')
    py.plot(K,J,x,marker='o',label=l)
    py.legend()
    
    print("Objective function values Initialization:"+str(initial+1))
    print("K:"+str(K))
    print("J(K):"+str(J))
    print("\n")
    py.show()
    initial+=1
    x='orange'
    z=5
py.show()
