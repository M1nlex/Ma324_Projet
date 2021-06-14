import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

X=np.arange(-3,3,0.15)
Y=np.copy(X)
X1,X2=np.meshgrid(X,Y)
Fr=X1**2 -X2**2
h=plt.contour(X1,X2,Fr,200)
p=plt.scatter(1,1,color='red')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title("ligne de contour Fr(x1,x2)")

fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot_surface(X1,X2,Fr,cmap=cm.coolwarm,linewidth=0)
plt.title("Tracer fr(x1,x2)")
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('fr(x1,x2)')
plt.tight_layout()
plt.show()
