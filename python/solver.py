from scipy.sparse import linalg
import numpy as np
        
# %% Transport Equation    

def transport(mesh,data):

    dt = data['dt']
    U = [data['u0']]
    fId = data['fId']
    flux = data['flux']
    u = data['u0'].copy()
    end = round(data['tMax']/dt)
    
    # Converts to optimized sparse
    
    mesh.M = mesh.M.tocsr()
    mesh.Sx = mesh.Sx.T.tocsr()
    mesh.Sy = mesh.Sy.T.tocsr()
    
    # Precompute oundary conditions
    
    face = mesh.precompute(fId)
    
    # Solves with Euler scheme

    for i in range(end):
        
        # Update and oundary conditions
        
        F = flux(u)
        B = mesh.neumannVar(face,F)
        S = mesh.Sx.dot(F[0])+mesh.Sy.dot(F[1])
        
        # Solves the system
        
        u += dt*linalg.spsolve(mesh.M,S-B)
        U.append(u.copy())
        u = np.abs(u)
    
    U = np.array(U)
    return U

# %% Diffusion Equation    

def diffusion(mesh,data):
    
    k = data['k']
    bc = data['bc']
    dt = data['dt']
    U = [data['u0']]
    nId = data['nId']
    u = data['u0'].copy()
    end = round(data['tMax']/dt)
    
    # Boundary conditions
    
    A = mesh.M.copy()
    MK = (mesh.M-k*dt*mesh.K).tocsr()
    
    A[nId] = 0
    A[nId,nId] = 1
    A = A.tocsr()
    
    # Solves with Euler scheme

    for i in range(end):
        
        # Update and boundary conditions
        
        b = MK.dot(u)
        b[nId] = bc
        
        # Solves the system
        
        u = linalg.spsolve(A,b)
        U.append(u.copy())
    
    U = np.array(U)
    return U

# %% Advection Equation

def advection(mesh,data):
    
    a = data['a']
    k = data['k']
    nId = data['nId']
    fId = data['fId']
    bcDir = data['bcDir']
    bcNeu = data['bcNeu']
    A = a[0]*mesh.Sx+a[1]*mesh.Sy+k*mesh.K
    b = mesh.F
    
    # Boundary conditions
    
    face = mesh.precompute(fId)
    B = mesh.neumannFix(face,bcNeu)
    
    A[nId] = 0
    B[nId] = 0
    b[nId] = bcDir
    A[nId,nId] = 1
    A = A.tocsr()
    
    # Solves the system
    
    u = linalg.spsolve(A,b+B)
    return u

# %% Poisson Equation

def laplace(mesh,data):
    
    bc = data['bc']
    nId = data['nId']

    # Boundary conditions
    
    A = mesh.K
    b = mesh.F
    
    A[nId] = 0
    b[nId] = bc
    A[nId,nId] = 1
    A = A.tocsr()
    
    # Solves the system
    
    u = linalg.spsolve(A,b)
    return u