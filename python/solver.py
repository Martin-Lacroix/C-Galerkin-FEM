from scipy import sparse
import numpy as np

# %% Solver

def solver(mesh,tVec,flux,u0):
    
    tStep = [0]
    dt = tVec[1]
    u = u0.copy()
    uStep = [u0]
    end = round(tVec[0]/dt)
    
    # Solves with Euler scheme

    for i in range(end):
        
        fx = flux[0](u)
        fy = flux[1](u)

        F = mesh.flux(fx,fy)
        S = mesh.Sx.dot(fx)+mesh.Sy.dot(fy)
        u += dt*sparse.linalg.spsolve(mesh.M,S-F)
        
        u = np.abs(u)
        uStep.append(u.copy())
        tStep.append((i+1)*dt)
    
    uStep = np.array(uStep)
    tStep = np.array(tStep)
    return uStep,tStep