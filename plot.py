from matplotlib import pyplot as plt
import numpy as np

# %% Read File

u = np.genfromtxt("output/solution.txt",delimiter=',')
nXY = np.genfromtxt("output/nXY.txt",delimiter=',')

nNbr = nXY.shape[0]
size = np.sqrt(nNbr).astype(int)
dx = np.abs(nXY[0,0]-nXY[1,0])
    
# %% Plots Solution scalar

cmap = 'Spectral_r'
plane = (1,1,1,0)
plt.rcParams['font.size'] = 14
grid = {'color':[0.9,0.9,0.9],'linewidth':0.5,'linestyle':'-'}

u = np.atleast_2d(u)
nbr = u.shape[0]
step = 40
k = 0

# %% Builds the scalar field

if u.shape[1]==nNbr:
    
    z = np.zeros((size,size))
    [z1,z2] = [np.min(u[0]),np.max(u[0])]
    x = np.linspace(np.min(nXY[:,0]),np.max(nXY[:,0]),size)
    y = np.linspace(np.max(nXY[:,1]),np.min(nXY[:,1]),size)
    x,y = np.meshgrid(x,y)
    
    # Plots the solution
    
    for i in range(0,nbr,int(nbr/step+1)):
        for j in range(size**2): z[tuple(np.rint(nXY[j]/dx).astype(int))] = u[i,j]
        z = np.flip(z.T,axis=0)
        
        fig = plt.figure(figsize=(12,6))
        ax = fig.gca(projection='3d')
        ax.set_xlabel('$x$',labelpad=20)
        ax.set_ylabel('$y$',labelpad=15)
        ax.set_zlabel('$u\,(x,y)$',labelpad=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.zaxis.set_major_locator(plt.MaxNLocator(4))
        ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap=cmap,vmin=z1,vmax=z2)
        ax.xaxis._axinfo['grid'] = grid
        ax.yaxis._axinfo['grid'] = grid
        ax.zaxis._axinfo['grid'] = grid
        ax.xaxis.set_pane_color(plane)
        ax.yaxis.set_pane_color(plane)
        ax.zaxis.set_pane_color(plane)
        ax.set_zlim(z1,z2)
        plt.tight_layout()
        
        plt.savefig("output/"+str(k)+".png",bbox_inches="tight")
        plt.show()
        plt.close()
        k += 1
    
# %% Builds the vector field

if u.shape[1]>nNbr:

    u1 = u[0,:nXY.shape[0]]
    u2 = u[0,nXY.shape[0]:]
    z1 = np.zeros((size,size))
    z2 = np.zeros((size,size))
    
    for i in range(size**2): z1[tuple(np.rint(nXY[i]/dx).astype(int))] = u1[i]
    for i in range(size**2): z2[tuple(np.rint(nXY[i]/dx).astype(int))] = u2[i]
    z1 = np.flip(z1.T,axis=0)
    z2 = np.flip(z2.T,axis=0)
    
    ext = [np.min(nXY[:,0]),np.max(nXY[:,0]),np.min(nXY[:,1]),np.max(nXY[:,1])]
    
    # Plots the solution ux
    
    plt.figure(figsize=(12,4))
    plt.imshow(z1,cmap=cmap,extent=ext,interpolation='spline16')
    plt.colorbar(label='$u\,_1(x,y)$')
    plt.tight_layout()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    
    plt.savefig("output/0.png",bbox_inches="tight")
    plt.show()
    plt.close()
    
    # Plots the solution uy
    
    plt.figure(figsize=(12,4))
    plt.imshow(z2,cmap=cmap,extent=ext,interpolation='spline16')
    plt.colorbar(label='$u\,_2\,(x,y)$')
    plt.tight_layout()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    
    plt.savefig("output/1.png",bbox_inches="tight")
    plt.show()
    plt.close()