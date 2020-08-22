from solver import advection
from mesh import Mesh
import numpy as np
import time
import os

# %% Functions

def build(elem,size,xyMax):
    
    # Node coordinates
    
    eId = []
    node = size+1
    nXY = np.zeros((node**2,2))
    idx = [i for i in range(node-1,node**2-node-1,node)]
    for i in range(node): nXY[i*node:(i+1)*node] = [[j,i] for j in range(node)]
    nXY *= xyMax/size
    
    # Element indices
    
    for i in range(node**2-node-1):
        if (i not in idx) and (elem==4): eId += [[i,i+1,i+node+1,i+node]]
        if (i not in idx) and (elem==3): eId += [[i,i+1,i+node]]+[[i+1,i+node+1,i+node]]

    # Face and node indices for BC
        
    x1 = np.arange(size,node**2,node)
    x2 = np.arange(node**2-2,node*size,-1)
    x3 = np.concatenate((x1[1:],x2,[x2[-1]-1]))
    fId = np.transpose([np.concatenate((x1,x2)),x3])
    
    x1 = np.arange(node)
    x2 = np.arange(0,node**2,node)[1:]
    nId = np.concatenate((x1,x2),axis=0)
    return nXY,eId,fId,nId
        
# %% Mesh And Flux

elem = 4
xyMax = 1
size = 50

fun = lambda x,y: x**0
nXY,eId,fId,nId = build(elem,size,xyMax)

data = {}
data['k'] = 1
data['a'] = [3,3]
data['bcDir'] = 0
data['fId'] = fId
data['nId'] = nId
data['bcNeu'] = -0.1

# %% Solver

start = time.time()
mesh = Mesh(nXY,eId,fun)
print('Mesh: ',time.time()-start,'[sec]')

start = time.time()
u = advection(mesh,data)
print('Solver: ',time.time()-start,'[sec]')

# %% Writes Solution

path = r'../output' 
if not os.path.exists(path): os.makedirs(path)
np.save('../output/solution.npy',u)
np.save('../output/nXY.npy',nXY)