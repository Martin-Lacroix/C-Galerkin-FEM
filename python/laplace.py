from solver import laplace
from mesh import Mesh
import numpy as np
import time
import os

# %% Functions

def meshparam(elem,size,xyMax):
    
    # Node coordinates
    
    eId = []
    node = size+1
    nXY = np.zeros((node**2,2))
    idx = [i for i in range(size,node**2-node-1,node)]
    for i in range(node): nXY[i*node:(i+1)*node] = [[j,i] for j in range(node)]
    nXY *= xyMax/size
    
    # Element indices
    
    for i in range(node**2-node-1):
        if (i not in idx):
            if (elem==4): eId += [[i,i+1,i+node+1,i+node]]
            if (elem==3): eId += [[i,i+1,i+node]]+[[i+1,i+node+1,i+node]]

    # Node indices for BC
    
    n1 = np.arange(node)
    n2 = np.arange(node,node**2,node)
    n3 = (np.arange(node)+size*node)[1:]
    n4 = (np.arange(node,node**2,node)-1)[1:]
    nId = np.sort(np.concatenate((n1,n2,n3,n4)))
    return nXY,eId,nId
        
# %% Mesh And Flux

elem = 3
xyMax = 1
size = 50

fun = lambda x,y: np.sin(2*x)+np.sin(2*y)
nXY,eId,nId = meshparam(elem,size,xyMax)

data = {}
data['bc'] = 0
data['nId'] = nId

# %% Solver

start = time.time()
mesh = Mesh(nXY,eId,fun)
print('Mesh:',time.time()-start,'[sec]')

start = time.time()
u = laplace(mesh,data)
print('Solver:',time.time()-start,'[sec]')

# %% Writes Solution

path = r'../output' 
if not os.path.exists(path): os.makedirs(path)
np.save('../output/solution.npy',u)
np.save('../output/nXY.npy',nXY)