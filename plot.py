from matplotlib import pyplot as plt
from scipy import interpolate
import numpy as np

# %% Read File

path = r"build/output/"
u = np.genfromtxt(path+"solution.txt",delimiter=',')
nXY = np.genfromtxt(path+"nXY.txt",delimiter=',')

# Parameters

mk = 's'
plane = (1,1,1,0)
cmap = 'Spectral_r'
nNbr = nXY.shape[0]
plt.rcParams['font.size'] = 14
grid = {'color':[0.9,0.9,0.9],'linewidth':0.5,'linestyle':'-'}
u = np.atleast_2d(u)
nbr = u.shape[0]
step = 40
k = 0

# %% Builds the scalar field

if u.shape[1]==nNbr:

    z1 = np.min(u[0])
    z2 = np.max(u[0])
    x = np.unique(nXY[:,0])
    y = np.unique(nXY[:,1])
    X,Y = np.meshgrid(x,y)
    xy = np.transpose([X,Y]).reshape(-1,2)
    
    # Plots the solution
    
    for i in range(0,nbr,int(nbr/step+1)):
    
        Z = interpolate.griddata(nXY,u[i],xy)
        Z = Z.reshape(X.shape).T
        
        fig = plt.figure(figsize=(12,6))
        ax = fig.gca(projection='3d')
        ax.set_xlabel('$x$',labelpad=20)
        ax.set_ylabel('$y$',labelpad=15)
        ax.set_zlabel('$u(x,y)$',labelpad=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.zaxis.set_major_locator(plt.MaxNLocator(4))
        ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cmap,vmin=z1,vmax=z2)
        ax.xaxis._axinfo['grid'] = grid
        ax.yaxis._axinfo['grid'] = grid
        ax.zaxis._axinfo['grid'] = grid
        ax.xaxis.set_pane_color(plane)
        ax.yaxis.set_pane_color(plane)
        ax.zaxis.set_pane_color(plane)
        ax.set_zlim(z1,z2)
        plt.tight_layout()
        
        plt.savefig(path+str(k)+".png",bbox_inches="tight")
        plt.show()
        plt.close()
    k += 1
    
# %% Builds the vector field

if u.shape[1]>nNbr:

    u1 = u[0,:nNbr]
    u2 = u[0,nNbr:]
    
    # Plots the solution ux
    
    plt.figure(figsize=(5.5,4))
    plt.scatter(nXY[:,0],nXY[:,1],c=u1,cmap=cmap,marker=mk)
    plt.colorbar(label='$u_1(x,y)$')
    plt.tight_layout()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    
    plt.savefig(path+"0.png",bbox_inches="tight")
    plt.show()
    plt.close()
    
    # Plots the solution uy
    
    plt.figure(figsize=(5.5,4))
    plt.scatter(nXY[:,0],nXY[:,1],c=u2,cmap=cmap,marker=mk)
    plt.colorbar(label='$u_2(x,y)$')
    plt.tight_layout()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    
    plt.savefig(path+"1.png",bbox_inches="tight")
    plt.show()
    plt.close()