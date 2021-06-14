import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

X=np.arange(-3,3,0.15)#définition de l'axe x1
Y=np.copy(X)#définition de l'axe x2
X1,X2=np.meshgrid(X,Y)#création d'une représentation selon plusieurs axes
Fr=(X1-1)**2 +10*(X1**2-X2)**2#définition de la méthode
h=plt.contour(X1,X2,Fr,1000)#création des lignes de courant
p=plt.scatter(1,1,color='red')#affichage du point critique (1,1)
plt.xlim(-3,3)#limitations des axes
plt.ylim(-3,3)
plt.title("ligne de contour Fr(x1,x2)")

fig=plt.figure()
ax=fig.gca(projection='3d')#définition de la forme 3d de la fonction
ax.plot_surface(X1,X2,Fr,cmap=cm.coolwarm,linewidth=0)#affichage de la fonction
plt.title("Tracer fr(x1,x2)")
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('fr(x1,x2)')
plt.tight_layout()
plt.show()
