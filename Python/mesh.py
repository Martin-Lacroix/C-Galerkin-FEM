from scipy import sparse
import numpy as np

# %% Triangles

class Triangle:

    def __init__(self,nXY):
        
        self.M = 0
        self.Sx = 0
        self.Sy = 0
        self.nXY = nXY
        self.gRS = np.zeros((4,2))
        self.wei = np.array([-27,25,25,25])/96
        
        # Integration points coordinates
        
        self.gRS[0] = [1/3,1/3]
        self.gRS[1] = [0.6,0.2]
        self.gRS[2] = [0.2,0.6]
        self.gRS[3] = [0.2,0.2]
        
        # Computes the Jacobian matrix
        
        J11 = -self.nXY[0,0]+self.nXY[1,0]
        J12 = -self.nXY[0,1]+self.nXY[1,1]
        J21 = -self.nXY[0,0]+self.nXY[2,0]
        J22 = -self.nXY[0,1]+self.nXY[2,1]
        
        self.detJ = J11*J22-J12*J21
        self.invJ = np.array([[J22,-J12],[-J21,J11]])/self.detJ
        
        # Local shape functions
        
        drN = np.array([-1,1,0])
        dsN = np.array([-1,0,1])
        N = lambda r,s: np.array([1-r-s,r,s])
        
        self.dxN = np.repeat([drN*self.invJ[0,0]+dsN*self.invJ[0,1]],4,axis=0).T
        self.dyN = np.repeat([drN*self.invJ[1,0]+dsN*self.invJ[1,1]],4,axis=0).T
        self.N = N(*self.gRS.T)
        
        # Mass and stifness matrix
        
        self.M = np.dot(self.wei*self.detJ*self.N,self.N.T)
        self.Sx = np.dot(self.wei*self.detJ*self.dxN,self.N.T)
        self.Sy = np.dot(self.wei*self.detJ*self.dyN,self.N.T) 

# %% Quadrangles

class Quadrangle:

    def __init__(self,nXY):
        
        self.nXY = nXY
        self.gRS = np.zeros((9,2))
        self.wei = np.array([25,25,25,25,40,40,40,40,64])/81
        
        # Integration points coordinates
        
        self.gRS[0] = [-np.sqrt(3/5),-np.sqrt(3/5)]
        self.gRS[1] = [np.sqrt(3/5),-np.sqrt(3/5)]
        self.gRS[2] = [-np.sqrt(3/5),np.sqrt(3/5)]
        self.gRS[3] = [np.sqrt(3/5),np.sqrt(3/5)]
        self.gRS[4] = [0,-np.sqrt(3/5)]
        self.gRS[5] = [-np.sqrt(3/5),0]
        self.gRS[6] = [np.sqrt(3/5),0]
        self.gRS[7] = [0,np.sqrt(3/5)]
        self.gRS[8] = [0,0]
        
        # Computes the Jacobian matrix
        
        J11 = lambda s: ((s-1)*self.nXY[0,0]+(1-s)*self.nXY[1,0]+(s+1)*self.nXY[2,0]-(s+1)*self.nXY[3,0])/4
        J12 = lambda s: ((s-1)*self.nXY[0,1]+(1-s)*self.nXY[1,1]+(s+1)*self.nXY[2,1]-(s+1)*self.nXY[3,1])/4
        J21 = lambda r: ((r-1)*self.nXY[0,0]-(r+1)*self.nXY[1,0]+(r+1)*self.nXY[2,0]+(1-r)*self.nXY[3,0])/4
        J22 = lambda r: ((r-1)*self.nXY[0,1]-(r+1)*self.nXY[1,1]+(r+1)*self.nXY[2,1]+(1-r)*self.nXY[3,1])/4
        detJ = lambda r,s: J11(s)*J22(r)-J12(s)*J21(r)
        
        invJ11 = lambda r,s: J22(r)/detJ(r,s)
        invJ12 = lambda r,s: -J12(r)/detJ(r,s)
        invJ21 = lambda r,s: -J21(r)/detJ(r,s)
        invJ22 = lambda r,s: J11(r)/detJ(r,s)
        
        # Local shape functions
        
        N = lambda r,s: np.array([(1-r)*(1-s)/4,(1+r)*(1-s)/4,(1+r)*(1+s)/4,(1-r)*(1+s)/4])
        drN = lambda s: np.array([(s-1)/4,(1-s)/4,(s+1)/4,-(s+1)/4])
        dsN = lambda r: np.array([(r-1)/4,-(r+1)/4,(r+1)/4,(1-r)/4])
        dxN = lambda r,s: drN(s)*invJ11(r,s)+dsN(r)*invJ12(r,s)
        dyN = lambda r,s: drN(s)*invJ21(r,s)+dsN(r)*invJ22(r,s)
        
        self.detJ = detJ(*self.gRS.T)
        self.dxN = dxN(*self.gRS.T)
        self.dyN = dyN(*self.gRS.T)
        self.N = N(*self.gRS.T)
        
        # Mass and stifness matrix
        
        self.M = np.dot(self.wei*self.detJ*self.N,self.N.T)
        self.Sx = np.dot(self.wei*self.detJ*self.dxN,self.N.T)
        self.Sy = np.dot(self.wei*self.detJ*self.dyN,self.N.T)  
        
# %% Faces

class Face:
    
    def __init__(self,nXY):
        
        self.M = 0
        self.nXY = nXY
        self.wei = np.array([5/9,5/9,8/9])
        self.gRS = np.array([-np.sqrt(3/5),np.sqrt(3/5),0])
        
        # Outer normal
        
        v = nXY[1]-nXY[0]
        self.norm = np.array([v[1],v[0]])
        self.detJ = np.sqrt(np.sum(v**2))/2
        self.norm /= np.sqrt(np.sum(self.norm**2))
        
        # Mass and local shape functions
        
        self.N = lambda r: np.array([(1-r)/2,(1+r)/2])
        M = lambda r: np.tensordot(self.N(r),self.N(r),axes=0)*self.detJ
        for i in range(3): self.M += self.wei[i]*M(self.gRS[i])
        
    def flux(self,fx,fy):
        
        F = np.abs(self.norm[0]*fx)+np.abs(self.norm[1]*fy)
        F = self.M.dot(F)
        return F

# %% Global Mesh

class Mesh:

    def __init__(self,nXY,eId,fId):
        
        self.nXY = np.array(nXY)
        self.eId = np.array(eId)
        self.fId = np.array(fId)
        
        self.nNbr = self.nXY.shape[0]
        self.eNbr = self.eId.shape[0]
        self.fNbr = self.fId.shape[0]
        
        self.M = sparse.dok_matrix((self.nNbr,self.nNbr))
        self.Sx = sparse.dok_matrix((self.nNbr,self.nNbr))
        self.Sy = sparse.dok_matrix((self.nNbr,self.nNbr))
        self.elems = []
        
        # Creates the boundary faces
        
        self.faces = [Face(self.nXY[self.fId[i]]) for i in range(self.fId.shape[0])]
        self.nfId = np.unique(self.fId.flatten())

        # Creates the domain elements
        
        for i in range(self.eNbr):
            
            if len(self.eId[i])==4:
                
                self.elems.append(Quadrangle(self.nXY[self.eId[i]]))
                
                # Adds to the global mass matrix
                
                self.M[self.eId[i][0],self.eId[i]] += self.elems[i].M[0]
                self.M[self.eId[i][1],self.eId[i]] += self.elems[i].M[1]
                self.M[self.eId[i][2],self.eId[i]] += self.elems[i].M[2]
                self.M[self.eId[i][3],self.eId[i]] += self.elems[i].M[3]
                
                # Adds to the global stifness matrices
        
                self.Sx[self.eId[i][0],self.eId[i]] += self.elems[i].Sx[0]
                self.Sx[self.eId[i][1],self.eId[i]] += self.elems[i].Sx[1]
                self.Sx[self.eId[i][2],self.eId[i]] += self.elems[i].Sx[2]
                self.Sx[self.eId[i][3],self.eId[i]] += self.elems[i].Sx[3]
                
                self.Sy[self.eId[i][0],self.eId[i]] += self.elems[i].Sy[0]
                self.Sy[self.eId[i][1],self.eId[i]] += self.elems[i].Sy[1]
                self.Sy[self.eId[i][2],self.eId[i]] += self.elems[i].Sy[2]
                self.Sy[self.eId[i][3],self.eId[i]] += self.elems[i].Sy[3]
                
            if len(self.eId[i])==3:
                
                self.elems.append(Triangle(self.nXY[self.eId[i]]))
                
                # Adds to the global mass matrix
                
                self.M[self.eId[i][0],self.eId[i]] += self.elems[i].M[0]
                self.M[self.eId[i][1],self.eId[i]] += self.elems[i].M[1]
                self.M[self.eId[i][2],self.eId[i]] += self.elems[i].M[2]
                
                # Adds to the global stifness matrices
        
                self.Sx[self.eId[i][0],self.eId[i]] += self.elems[i].Sx[0]
                self.Sx[self.eId[i][1],self.eId[i]] += self.elems[i].Sx[1]
                self.Sx[self.eId[i][2],self.eId[i]] += self.elems[i].Sx[2]
                
                self.Sy[self.eId[i][0],self.eId[i]] += self.elems[i].Sy[0]
                self.Sy[self.eId[i][1],self.eId[i]] += self.elems[i].Sy[1]
                self.Sy[self.eId[i][2],self.eId[i]] += self.elems[i].Sy[2]
        
        # Converts to optimized sparse
        
        self.Sx = self.Sx.tocsr()
        self.Sy = self.Sy.tocsr()
        self.M = self.M.tocsr()
        
    def flux(self,fx,fy):
        
        F = np.zeros(self.nNbr)
        for i in range(self.fNbr): F[self.fId[i]] += self.faces[i].flux(fx[self.fId[i]],fy[self.fId[i]])
        return F