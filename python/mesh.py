from scipy import sparse
import numpy as np

# %% 2D Triangles

class Triangle:

    def __init__(self,nXY,fun=0):
        
        gRS = np.zeros((4,2))
        wei = np.array([-27,25,25,25])/96
        
        # Integration points coordinates
        
        gRS[0] = [1/3,1/3]
        gRS[1] = [0.6,0.2]
        gRS[2] = [0.2,0.6]
        gRS[3] = [0.2,0.2]
        
        # Computes the Jacobian matrix
        
        J11 = -nXY[0,0]+nXY[1,0]
        J12 = -nXY[0,1]+nXY[1,1]
        J21 = -nXY[0,0]+nXY[2,0]
        J22 = -nXY[0,1]+nXY[2,1]
        
        detJ = J11*J22-J12*J21
        invJ = np.array([[J22,-J12],[-J21,J11]])/detJ
        
        # Integration rule
        
        drN = np.array([-1,1,0])
        dsN = np.array([-1,0,1]) 
        dxN = np.repeat([drN*invJ[0,0]+dsN*invJ[0,1]],4,axis=0).T
        dyN = np.repeat([drN*invJ[1,0]+dsN*invJ[1,1]],4,axis=0).T
        N = lambda r,s: np.array([1-r-s,r,s])
        
        N = N(*gRS.T)
        [x,y] = np.dot(N.T,nXY).T
        if callable(fun): F = fun(x,y)
        
        # Mass and stifness matrix
        
        self.M = np.dot(wei*detJ*N,N.T)
        self.Sx = np.dot(wei*detJ*N,dxN.T)
        self.Sy = np.dot(wei*detJ*N,dyN.T)
        if callable(fun): self.F = np.dot(wei*detJ*N,F)
        self.K = np.dot(wei*detJ*dxN,dxN.T)+np.dot(wei*detJ*dyN,dyN.T)

# %% 2D Quadrangles

class Quadrangle:

    def __init__(self,nXY,fun=0):
        
        gRS = np.zeros((9,2))
        wei = np.array([25,25,25,25,40,40,40,40,64])/81
    
        # Integration points coordinates
        
        gRS[0] = [-np.sqrt(3/5),-np.sqrt(3/5)]
        gRS[1] = [np.sqrt(3/5),-np.sqrt(3/5)]
        gRS[2] = [-np.sqrt(3/5),np.sqrt(3/5)]
        gRS[3] = [np.sqrt(3/5),np.sqrt(3/5)]
        gRS[4] = [0,-np.sqrt(3/5)]
        gRS[5] = [-np.sqrt(3/5),0]
        gRS[6] = [np.sqrt(3/5),0]
        gRS[7] = [0,np.sqrt(3/5)]
        gRS[8] = [0,0]
        
        # Computes the Jacobian matrix
        
        J11 = lambda s: ((s-1)*nXY[0,0]+(1-s)*nXY[1,0]+(s+1)*nXY[2,0]-(s+1)*nXY[3,0])/4
        J12 = lambda s: ((s-1)*nXY[0,1]+(1-s)*nXY[1,1]+(s+1)*nXY[2,1]-(s+1)*nXY[3,1])/4
        J21 = lambda r: ((r-1)*nXY[0,0]-(r+1)*nXY[1,0]+(r+1)*nXY[2,0]+(1-r)*nXY[3,0])/4
        J22 = lambda r: ((r-1)*nXY[0,1]-(r+1)*nXY[1,1]+(r+1)*nXY[2,1]+(1-r)*nXY[3,1])/4
        detJ = lambda r,s: J11(s)*J22(r)-J12(s)*J21(r)
        
        invJ11 = lambda r,s: J22(r)/detJ(r,s)
        invJ12 = lambda r,s: -J12(r)/detJ(r,s)
        invJ21 = lambda r,s: -J21(r)/detJ(r,s)
        invJ22 = lambda r,s: J11(r)/detJ(r,s)
        
        # Integration rule
        
        x = nXY[0,0]+(nXY[1,0]-nXY[0,0])*gRS[:,0]+(nXY[2,0]-nXY[0,0])*gRS[:,1]
        y = nXY[0,1]+(nXY[1,1]-nXY[0,1])*gRS[:,0]+(nXY[2,1]-nXY[0,1])*gRS[:,1]
        if callable(fun): F = fun(x,y)
        
        N = lambda r,s: np.array([(1-r)*(1-s)/4,(1+r)*(1-s)/4,(1+r)*(1+s)/4,(1-r)*(1+s)/4])
        drN = lambda s: np.array([(s-1)/4,(1-s)/4,(s+1)/4,-(s+1)/4])
        dsN = lambda r: np.array([(r-1)/4,-(r+1)/4,(r+1)/4,(1-r)/4])
        dxN = lambda r,s: drN(s)*invJ11(r,s)+dsN(r)*invJ12(r,s)
        dyN = lambda r,s: drN(s)*invJ21(r,s)+dsN(r)*invJ22(r,s)
        
        N = N(*gRS.T)
        dxN = dxN(*gRS.T)
        dyN = dyN(*gRS.T)
        detJ = detJ(*gRS.T)
        [x,y] = np.dot(N.T,nXY).T
        if callable(fun): F = fun(x,y)
        
        # Mass and stifness matrix
        
        self.M = np.dot(wei*detJ*N,N.T)
        self.Sx = np.dot(wei*detJ*N,dxN.T)
        self.Sy = np.dot(wei*detJ*N,dyN.T)
        if callable(fun): self.F = np.dot(wei*detJ*N,F)
        self.K = np.dot(wei*detJ*dxN,dxN.T)+np.dot(wei*detJ*dyN,dyN.T)

# %% 1D Line
        
class Line:

    def __init__(self,nXY):
        
        gRS = np.array([-np.sqrt(3/5),np.sqrt(3/5),0])
        detJ = np.sqrt(np.sum((nXY[1]-nXY[0])**2))/2
        N = np.array([(1-gRS)/2,(1+gRS)/2])
        wei = np.array([5/9,5/9,8/9])
        
        # Computes matrices
        
        self.norm = np.flip(nXY[1]-nXY[0])/(2*detJ)
        self.N = np.sum(wei*detJ*N,axis=1)
        self.M = np.dot(wei*detJ*N,N.T)

# %% Global Mesh

class Mesh:

    def __init__(self,nXY,eId,fun=0):
        
        self.nXY = np.array(nXY)
        self.eId = np.array(eId)
        
        self.nNbr = self.nXY.shape[0]
        self.eNbr = self.eId.shape[0]
        self.type = self.eId.shape[1]
        
        self.F = np.zeros(self.nNbr)
        self.M = sparse.dok_matrix((self.nNbr,self.nNbr))
        self.K = sparse.dok_matrix((self.nNbr,self.nNbr))
        self.Sx = sparse.dok_matrix((self.nNbr,self.nNbr))
        self.Sy = sparse.dok_matrix((self.nNbr,self.nNbr))

        # Computes global matrices

        for i in range(self.eNbr):
           
            if(self.type==4): elem = Quadrangle(self.nXY[self.eId[i]],fun)
            if(self.type==3): elem = Triangle(self.nXY[self.eId[i]],fun)
            if callable(fun): self.F[self.eId[i]] += elem.F
             
            for j in range(self.type):
         
                 self.M[self.eId[i][j],self.eId[i]] += elem.M[j]
                 self.K[self.eId[i][j],self.eId[i]] += elem.K[j]
                 self.Sx[self.eId[i][j],self.eId[i]] += elem.Sx[j]
                 self.Sy[self.eId[i][j],self.eId[i]] += elem.Sy[j]
                 
    # Precompute Neumann BC
                 
    def precompute(self,fId):

        face = [Line(self.nXY[nodes]) for nodes in fId]
        return face
    
    # Apply Neumann BC

    def neumannFix(self,fId,face,bc):
        
        B = np.zeros(self.nNbr)
        for i in range(len(face)): B[fId[i]] += face[i].N*bc
        return B
    
    # Apply Neumann BC
    
    def neumannVar(self,fId,face,F):
        
        Fx,Fy = F
        B = np.zeros(self.nNbr)
        for i in range(len(face)):
            
            nx,ny = face[i].norm
            nf = np.abs(nx*Fx[fId[i]])+np.abs(ny*Fy[fId[i]])
            B[fId[i]] += np.dot(face[i].M,nf)
        
        return B