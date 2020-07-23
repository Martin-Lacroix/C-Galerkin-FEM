from matplotlib import pyplot as plt
from solver import solver
from scipy import stats
from mesh import Mesh
import numpy as np
import time
import os

# %% Functions

def build(elem,size,dx):
    
    # Node coordinates
    
    eId = []
    nXY = np.zeros((size**2,2))
    fId = np.zeros((4*(size-1),2)).astype(int)
    idx = [i for i in range(size-1,size**2-size-1,size)]
    for i in range(size): nXY[i*size:(i+1)*size] = [[j,i] for j in range(size)]
    nXY *= dx
    
    # Element indices Q4
    
    if elem==4:
    
        for i in range(size**2-size-1):
            if i not in idx: eId.append([i,i+1,i+size+1,i+size])
    
    # Element indices T3
    
    if elem==3:
        
        for i in range(size**2-size-1):
            if i not in idx:
                eId.append([i,i+1,i+size])
                eId.append([i+1,i+size+1,i+size])

    # Boundary face indices
    
    for i in range(size-1):
        
        fId[i] = [i,i+1]
        fId[(size-1)+i] = [(i+1)*size-1,(i+2)*size-1]
        fId[2*(size-1)+i] = [size**2-i-1,size**2-i-2]
        fId[3*(size-1)+i] = [(size-1-i)*size,(size-2-i)*size]
        
    return nXY,eId,fId
    
def gaussian(nXY,dx,center):
    
    nbr = nXY.shape[0]
    [x,y] = np.array(center)*dx
        
    u = np.zeros(nbr)
    pdfx = stats.norm(x,1).pdf
    pdfy = stats.norm(y,1).pdf
    joint = lambda x,y: pdfx(x)*pdfy(y)
    
    for i in range(nbr): u[i] = joint(*tuple(nXY[i]))
    return u/np.max(u)
        
# %% Mesh And Flux

dx = 0.2
elem = 4
size = 60
tVec = [1,0.001]
center = [30,30]

def fx(u): return 6*u
def fy(u): return -6*u

nXY,eId,fId = build(elem,size,dx)
u0 = gaussian(nXY,dx,center)
flux = [fx,fy]
uStep = []

# %% Solver

start = time.time()
mesh = Mesh(nXY,eId,fId)
print('Mesh: ',time.time()-start)

start = time.time()
uStep,tStep = solver(mesh,tVec,flux,u0)
print('Solver: ',time.time()-start)

# %% Writes Solution

path = r'../output' 
if not os.path.exists(path): os.makedirs(path)

np.save('../output/solution.npy',uStep)
np.save('../output/time.npy',tStep)
np.save('../output/nXY.npy',nXY)