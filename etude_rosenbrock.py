import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

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


def gradient_pas_fixe(f, grad, x0, alpha, tol, imax=10 ** 6):
    x = x0
    xit = []
    nit = 0
    while (np.linalg.norm(grad(x)) > tol) and nit <= imax:
        nit += 1
        xit.append(x)
        x = x - alpha * grad(x)

    return x, np.asarray(xit).T, nit


def conditionOptimalite(Xnum, eps):
    g = rosenbrock_grad(Xnum)
    l = [(102-np.sqrt(10244))/2, (102+np.sqrt(10244))/2]
    test = 0
    for i in l:
        if i > 0:
            test += 1
    if np.linalg.norm(g) <= eps and test == len(l):
        mini = 1
    else:
        mini = 0
    return g, l, mini

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

        print(nit)

    return x, np.asarray(xit).T, nit
if __name__ == '__main__':
    # x, xit, nit = gradient_pas_fixe(rosenbrock, rosenbrock_grad, ((-1, 2), (1, 1)), 10 ** -3, 10 ** -6, 10 ** 6)
    # print(x, nit)
    # print(conditionOptimalite(x, 10**-4))
    #
    x, xit, nit = gradient_pas_fixe(rosenbrock, rosenbrock_grad, ((1, 2),(1,2)),10**-4, 10 ** -6, 10 ** 6)
    print(x, nit)
    print(conditionOptimalite(x, 10 ** -4))

    l = []
    t = []
    j = 0

    l1 = []
    t1 = []

    for i in xit.T:
        t.append(j)
        l.append(rosenbrock(i)[1])
        j += 1
    x, xit, nit = gradient_pas_optimal(rosenbrock, rosenbrock_grad, (1, 2), 10 ** -6, 10 ** 6)
    print(x, nit)
    j = 0
    for i in xit.T:
        t1.append(j)
        l1.append(rosenbrock(i))
        j += 1




    plt.semilogx(t, l, label="Pas Fixe")
    plt.semilogx(t1, l1, label="Pas Optimal")
    plt.title("Valeur de f en fonction des d’itérations")
    plt.legend()
    plt.show()
    # step = 0.1
    # a = -3.
    # b = 3.1  # extreme points in the x-axis
    # c = -3
    # d = 3.1  # extreme points in the y-axis
    # x, y = np.meshgrid(np.arange(a, b, step), np.arange(c, d, step))  # Création du pavé
    #
    # z = rosenbrock((x, y))
    #
    # CS = plt.contour(x, y, z, np.linspace(0, 10, 15))
    # plt.title(r'$\mathrm{Rosenbrock \ gradient \ optimal}$')
    # plt.xlabel('x')
    # plt.ylabel('y')
    #
    # plt.plot(xit[0], xit[1],)
    #
    # plt.show()
