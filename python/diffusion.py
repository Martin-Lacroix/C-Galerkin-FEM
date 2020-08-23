from solver import diffusion
from scipy import stats
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
    
def gaussian(nXY,xyMax):
    
    nNbr = nXY.shape[0]
    pdf = stats.norm(xyMax/2,1).pdf
    joint = lambda x,y: pdf(x)*pdf(y)
    u = np.zeros(nNbr)
    
    for i in range(nNbr): u[i] = joint(*tuple(nXY[i]))
    return u/np.max(u)
        
# %% Mesh And Flux

elem = 3
xyMax = 10
size = 50

nXY,eId,nId = meshparam(elem,size,xyMax)
u0 = gaussian(nXY,xyMax)

data = {}
data['k'] = 1
data['bc'] = 0
data['u0'] = u0
data['tMax'] = 1
data['nId'] = nId
data['dt'] = 0.001

# %% Solver

start = time.time()
mesh = Mesh(nXY,eId)
print('Mesh: ',time.time()-start,'[sec]')

start = time.time()
u = diffusion(mesh,data)
print('Solver: ',time.time()-start,'[sec]')

# %% Writes Solution

path = r'../output' 
if not os.path.exists(path): os.makedirs(path)
np.save('../output/solution.npy',u)
np.save('../output/nXY.npy',nXY)