from matplotlib import pyplot as plt
import numpy as np

# %% Read File

u = np.genfromtxt("output/disp.txt",delimiter=',')
nXY = np.genfromtxt("output/nXY.txt",delimiter=',')
e = np.genfromtxt("output/strain.txt",delimiter=',')

# Parameters

mk = 's'
cmap = 'Spectral_r'
plt.rcParams['font.size'] = 14
nbr = nXY.shape[0]
    
# %% Builds the vector field

u1 = u[:nbr]
u2 = u[nbr:]
e1 = e[:nbr]
e12 = e[2*nbr:]
e2 = e[nbr:2*nbr]

# Plots the solution ux

plt.figure(figsize=(5.5,4))
plt.scatter(nXY[:,0],nXY[:,1],c=u1,cmap=cmap,marker=mk)
plt.colorbar(label='$u\,_1\,(x,y)$')
plt.tight_layout()
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.savefig("output/0.png",bbox_inches="tight")
plt.show()
plt.close()

# Plots the solution uy

plt.figure(figsize=(5.5,4))
plt.scatter(nXY[:,0],nXY[:,1],c=u2,cmap=cmap,marker=mk)
plt.colorbar(label='$u\,_2\,(x,y)$')
plt.tight_layout()
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.savefig("output/1.png",bbox_inches="tight")
plt.show()
plt.close()