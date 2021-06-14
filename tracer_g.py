import numpy as np
import matplotlib.pyplot as plt


def mk_g():
    def g(x):
        return x[0] ** 2 + x[1] ** 2 * (x[1] ** 2 + 2)

    return g


if __name__ == '__main__':
    q = mk_g()

    step = 0.1
    a = -3.
    b = 3.1  # extreme points in the x-axis
    c = -3
    d = 3.1  # extreme points in the y-axis
    x, y = np.meshgrid(np.arange(a, b, step), np.arange(c, d, step))  # Création du pavé

    z = q((x, y))

    ax3d = plt.axes(projection='3d')
    ax3d.plot_surface(x, y, z, cmap='plasma')
    ax3d.set_title(r'$\mathrm{Représentation \ de:g(X)=x_1^{2}+x_2^{2}x(x_2^{2}+2)}$')

    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    plt.show()

    CS = plt.contour(x, y, z, np.linspace(0, 10, 15))
    plt.title(r'$\mathrm{Lignes \ de \ niveau \ de:g(X)=x_1^{2}+x_2^{2}x(x_2^{2}+2)}$')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


