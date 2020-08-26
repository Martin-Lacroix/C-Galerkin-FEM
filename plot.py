from matplotlib import pyplot as plt
import numpy as np

# %% Read File

u = np.genfromtxt("output/solution.txt",delimiter=',')
nXY = np.genfromtxt("output/nXY.txt",delimiter=',')

size = np.sqrt(nXY.shape[0]).astype(int)
dx = np.abs(nXY[0,0]-nXY[1,0])
    
# %% Plots Solution

k = 0
step = 40
u = np.atleast_2d(u)
nbr = u.shape[0]

#  Builds the graphs
    
z = np.zeros((size,size))
zlim = [np.min(u[0]),np.max(u[0])]
x = np.linspace(np.min(nXY[:,0]),np.max(nXY[:,0]),size)
y = np.linspace(np.min(nXY[:,1]),np.max(nXY[:,1]),size)
x,y = np.meshgrid(x,y)

# Plots the solution

for i in range(0,nbr,int(nbr/step+1)):
    for j in range(size**2): z[tuple(np.rint(nXY[j]/dx).astype(int))] = u[i,j]
    
    fig = plt.figure(figsize=(12,6))
    ax = fig.gca(projection='3d')
    ax.set_xlabel('$y$',labelpad=20)
    ax.set_ylabel('$x$',labelpad=15)
    ax.set_zlabel('$u\,(x,y)$',labelpad=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.zaxis.set_major_locator(plt.MaxNLocator(4))
    ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap='coolwarm',vmin=zlim[0],vmax=zlim[1])
    ax.xaxis._axinfo['grid'] = {'color':[0.9,0.9,0.9],'linewidth':0.5,'linestyle':'-'}
    ax.yaxis._axinfo['grid'] = {'color':[0.9,0.9,0.9],'linewidth':0.5,'linestyle':'-'}
    ax.zaxis._axinfo['grid'] = {'color':[0.9,0.9,0.9],'linewidth':0.5,'linestyle':'-'}
    ax.xaxis.set_pane_color((1,1,1,0))
    ax.yaxis.set_pane_color((1,1,1,0))
    ax.zaxis.set_pane_color((1,1,1,0))
    ax.set_zlim(zlim[0],zlim[1])
    plt.tight_layout()
    
    plt.savefig("output/"+str(k)+".png",bbox_inches="tight",format="png")
    plt.show()
    plt.close()
    k += 1
