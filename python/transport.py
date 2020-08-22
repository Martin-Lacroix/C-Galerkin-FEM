from solver import transport
from scipy import stats
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

    # Face indices for BC
    
    x1 = np.arange(size)
    x2 = np.arange(size,node**2,node)
    x3 = np.arange(x2[-1]-1,x2[-1]-node,-1)
    x4 = np.flip(np.arange(node,size**2,node))
    x = np.concatenate((x1,x2,x3,x4),axis=0)
    fId = np.transpose([x,np.roll(x,-1)])
    return nXY,eId,fId
    
def gaussian(nXY,xyMax):
    
    nNbr = nXY.shape[0]
    pdf = stats.norm(xyMax/2,1).pdf
    joint = lambda x,y: pdf(x)*pdf(y)
    u = np.zeros(nNbr)
    
    for i in range(nNbr): u[i] = joint(*tuple(nXY[i]))
    return u/np.max(u)
        
# %% Mesh And Flux

elem = 4
size = 50
xyMax = 10

def flux(u): return [6*u,-6*u]
nXY,eId,fId = build(elem,size,xyMax)
u0 = gaussian(nXY,xyMax)

data = {}
data['u0'] = u0
data['tMax'] = 1
data['fId'] = fId
data['dt'] = 0.001
data['flux'] = flux

# %% Solver

start = time.time()
mesh = Mesh(nXY,eId)
print('Mesh:',time.time()-start,'[sec]')

start = time.time()
u = transport(mesh,data)
print('Solver:',time.time()-start,'[sec]')

# %% Writes Solution

path = r'../output' 
if not os.path.exists(path): os.makedirs(path)
np.save('../output/solution.npy',u)
np.save('../output/nXY.npy',nXY)