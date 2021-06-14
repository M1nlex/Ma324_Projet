import numpy as np
import matplotlib.pyplot as plt


def mk_q(n):
    def q(x):
        s0, s1, s2 = 0, 0, 0
        for i in range(n):
            s0 += x[i] ** 2
            s2 += x[i]
        for i in range(n - 1):
            s1 += x[i] * x[i + 1]
        return s0 - s1 - s2

    return q


if __name__ == '__main__':
    q = mk_q(2)

    step = 0.1
    a = -5.
    b = 5.1  # extreme points in the x-axis
    c = -5
    d = 5.1  # extreme points in the y-axis
    x, y = np.meshgrid(np.arange(a, b, step), np.arange(c, d, step))  # Création du pavé

    z = q((x, y))

    ax3d = plt.axes(projection='3d')
    ax3d.plot_surface(x, y, z, cmap='plasma')
    ax3d.set_title(r'$\mathrm{Représentation \ de:q_n(X)=\sum_{i=1}^{n} x_i^{2} - \sum_{i=1}^{n-1} x_ix_{i+1} - \sum_{'
                   r'i=1}^{n} x_i}$')
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    plt.show()

    CS = plt.contour(x, y, z, np.linspace(0, 10, 15))

    plt.title(r'$\mathrm{Lignes \ de \ niveau \ de:q_n(X)=\sum_{i=1}^{n} x_i^{2} - \sum_{i=1}^{n-1} x_ix_{i+1} - \sum_{'
              r'i=1}^{n} x_i}$')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()
