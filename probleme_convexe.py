import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.optimize as op


def mk_q(n):
    def q(x):
        s0, s1, s2 = 0, 0, 0
        for i in range(n):
            s0 += x[i] ** 2
            s2 += x[i]
        for i in range(n - 1):
            s1 += x[i] * x[i + 1]
        return s0 - s1 - s2

    def q_grad(x):
        grad = np.zeros(n)
        for i in range(len(x)):
            s0, s1, s2 = 0, 0, 0
            for j in range(n):
                if j != i:
                    s0 += x[j] ** 2
                    s2 += x[j]
                else:
                    s0 += 2 * x[j]
                    s2 += 1
            for j in range(n - 1):
                if j == i - 1:
                    s1 += x[j]
                elif j == i:
                    s1 += x[j + 1]
                else:
                    s1 += x[j] * x[j + 1]

            grad[i] = s0 - s1 - s2
        return grad

    return q, q_grad


def rosenbrock(x):
    y = np.asarray(x)
    return (y[0] - 1) ** 2 + 10 * (y[0] ** 2 - y[1]) ** 2


def rosenbrock_grad(x):
    y = np.asarray(x)
    grad = np.zeros_like(y)
    grad[0] = 2 * y[0] - 2 + 40 * (y[0] ** 3 - y[0] * y[1])
    grad[1] = 20 * (y[1] - y[0] ** 2)
    return grad


def rosenbrock_hessian_(x):
    y = np.asarray(x)
    return np.array((
        (2 + 10 * (12 * y[0] ** 2 - 4 * y[1]), -40 * y[0]),
        (-40 * y[0], 20)))


def mk_g():
    def g(x):
        return x[0] ** 2 + x[1] ** 2 * (x[1] ** 2 + 2)

    def grad_g(x):
        return np.array((2 * x[0], 4 * x[1] ** 3 + 4 * x[1]))

    def hessg(x):
        return np.array([[2, 0], [0, 12 * (x[1] ** 2) + 4]])

    return g, grad_g, hessg


def gradient_pas_fixe(f, grad, x0, alpha, tol, imax=10 ** 6):
    x = x0
    xit = []
    nit = 0
    while (np.linalg.norm(grad(x)) > tol) and nit <= imax:
        nit += 1
        xit.append(x)
        x = x - alpha * grad(x)

    return x, np.asarray(xit).T, nit


def gradient_pas_optimal(f, grad, x0, tol=10 ** -6, imax=10 ** 6):
    x = x0
    xit = []
    nit = 0
    d = -grad(x)

    def h(t):
        return f(x + t * d)

    while (np.linalg.norm(d) > tol) and nit <= imax:
        nit += 1
        xit.append(x)
        # alpha = newton(f, grad, x, d, 1)
        alpha = op.minimize(h, np.array([1]))['x']
        x = x + alpha * d
        d = -grad(x)

    return x, np.asarray(xit).T, nit


def conditionOptimalite(Xnum, eps):
    g = grad_g(Xnum)
    l = np.linalg.eig(hg(Xnum))[0]
    test = 0
    for i in l:
        if i > 0:
            test += 1
    if (np.linalg.norm(g) <= eps and test == len(l)):
        mini = 1
    else:
        mini = 0
    return g, l, mini


if __name__ == '__main__':

    g, grad_g, hg = mk_g()

    # alpha = (10 ** (-5) )
    # alpha = 1
    alpha = 0.01
    temps1 = time.time()
    # x, xit, nit = gradient_pas_fixe(g, grad_g, (2, 2), alpha, 10 ** -6, 10 ** 6)
    x, xit, nit = gradient_pas_optimal(g, grad_g, (2, 2), 10 ** -6, 10 ** 6)
    temps2 = time.time()

    print(conditionOptimalite(x, (10 ** (-4))))
    print(nit)
    print(temps2 - temps1)
    print(x)
    print(g(x))

    plt.plot(xit[0], xit[1])

    step = 0.1
    a = -5.
    b = 5.1  # extreme points in the x-axis
    c = -5
    d = 5.1  # extreme points in the y-axis
    x, y = np.meshgrid(np.arange(a, b, step), np.arange(c, d, step))  # Création du pavé

    z = g((x, y))

    CS = plt.contour(x, y, z, np.linspace(0, 10, 15))
    plt.title(r'$\mathrm{Lignes \ de \ niveau \ de:g(X)=x_1^{2}+x_2^{2}x(x_2^{2}+2)}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # print(xit)
    git = []
    it = []
    j = 0
    for i in range(0, nit):
        git.append(g([xit[0][i], xit[1][i]]))
        j += 1
        it.append(j)
    plt.semilogx(it, git)
    plt.title(r'$\mathrm{g(x^{*}) \ en \ fonction \ de \ l \ itération}$')
    plt.xlabel('itérations')
    plt.ylabel(r'$\mathrm{Valeurs \ de \ g(x^{*})}$')
    plt.show()
