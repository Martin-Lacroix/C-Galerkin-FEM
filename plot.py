from matplotlib import pyplot as plt
import numpy as np

# %% Read File

dx = 0.2
size = 60

try:
    uStep = np.genfromtxt("output/solution.txt",delimiter=',')
    time = np.genfromtxt("output/time.txt",delimiter=',')
    nXY = np.genfromtxt("output/nXY.txt",delimiter=',')
except:
    uStep = np.load("output/solution.txt")
    time = np.load("output/time.txt")
    nXY = np.load("output/nXY.txt")

# %% Display Solution

k = 0
step = 40
nbrStep = len(uStep)
for i in range(0,nbrStep,int(nbrStep/step)):

    u = uStep[i]
    x = np.arange(size)
    y = np.arange(size)
    x,y = np.meshgrid(x,y)
    z = np.zeros((size,size))
    for j in range(size**2): z[tuple(np.rint(nXY[j]/dx).astype(int))] = u[j]
    z = z.T

    # Plots figures

    fig = plt.figure(i,figsize=(12,6))
    ax = fig.gca(projection='3d')
    ax.set_xlabel('$y$',labelpad=20)
    ax.set_ylabel('$x$',labelpad=15)
    ax.set_zlabel('Value',labelpad=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.zaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_pane_color((1,1,1,0))
    ax.yaxis.set_pane_color((1,1,1,0))
    ax.zaxis.set_pane_color((1,1,1,0))
    ax.xaxis._axinfo['grid'] = {'color': [0.9,0.9,0.9], 'linewidth': 0.5, 'linestyle': '-'}
    ax.yaxis._axinfo['grid'] = {'color': [0.9,0.9,0.9], 'linewidth': 0.5, 'linestyle': '-'}
    ax.zaxis._axinfo['grid'] = {'color': [0.9,0.9,0.9], 'linewidth': 0.5, 'linestyle': '-'}
    ax.plot_surface(y,x,z,rstride=1,cstride=1,cmap='coolwarm',vmin=0,vmax=1)
    ax.set_zlim(-1,1)
    plt.tight_layout()
    
    plt.savefig("output/"+str(k)+".png",bbox_inches="tight",format="png",transparent=False)
    k += 1