import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import matplotlib.animation as animation
import matplotlib
matplotlib.use("Agg")

# all參數

fig, ax = plt.subplots()

lr = 0.00001
t = np.arange(-30, 60, 0.1)
x = -25
y = x ** 4 - 50 * (x ** 3) - x + 1
dy_dx = 4 * (x ** 3) - 150 * (x ** 2) - 1
epoch = 200
min = []
sgd_x = 0
sgd_dy_dx = 0
sgdX = []
sgdY = []

v = 0
beta = 0.9
momentum_x = 0
momentum_dy_dx = 0
momentumX = []
momentumY = []



def minimum_y():
    for i in t:
        out = i ** 2 - 5
        min.append(out)
    min_y = np.min(min)
    return min_y


def SGD(x, lr, dy_dx, sgd_x):
    if sgd_x == 0:
        x = x - lr * dy_dx
        return x
    else:
        sgd_x = sgd_x - lr * dy_dx
        return sgd_x


def sgd_plot(x, sgd_x, min_y):
    for i in range(epoch):
        if i == 0:
            sgd_x = SGD(x, lr, dy_dx, sgd_x)

        else:
            sgd_dy_dx = 4 * (sgd_x ** 3) - 150 * (sgd_x ** 2) - 1
            sgd_x = SGD(x, lr, sgd_dy_dx, sgd_x)
        y = sgd_x ** 4 - 50 * (sgd_x ** 3) - sgd_x + 1
        z = np.abs(y - min_y)
        if z < 0.01:
            break
        if i % 1 == 0:
            sgdX.append(sgd_x)
            sgdY.append(y)
    sgd_Plot = plt.scatter(sgdX, sgdY, marker='o', edgecolors='r', color='')
    return sgdX, sgd_Plot

def momentum(x, lr, v, beta, dy_dx, momentum_x):
    if momentum_x == 0:
        v = beta * v - lr * dy_dx
        x = x + v
        return x, v
    else:
        v = beta * v - lr * dy_dx
        momentum_x = momentum_x + v
        return momentum_x, v

def momentum_plot(x, momentum_x, min_y, v):
    for i in range(epoch):
        if i == 0:
            momentum_x, v = momentum(x, lr, v, beta, dy_dx, momentum_x)

        else:
            momentum_dy_dx = 4 * (momentum_x ** 3) - 150 * (momentum_x ** 2) - 1
            momentum_x, v = momentum(x, lr, v, beta, momentum_dy_dx, momentum_x)
        y = momentum_x ** 4 - 50 * (momentum_x ** 3) - momentum_x + 1
        z = np.abs(y - min_y)
        if z < 0.01:
            break
        if i % 1 == 0:
            momentumX.append(momentum_x)
            momentumY.append(y)
    momentum_Plot = plt.scatter(momentumX, momentumY, marker='o', edgecolors='g', color='')
    return momentumX, momentum_Plot

min_y = minimum_y()
sgd_X, sgd_PLOT = sgd_plot(x, sgd_x, min_y)
momentum_X, momentum_PLOT = momentum_plot(x, momentum_x, min_y, v)

def sgd_update(n, fig, scat1, scat2):
    scat1.set_offsets(([sgd_X[n], sgd_X[n] ** 4 - 50 * sgd_X[n] ** 3 - sgd_X[n] + 1]))
    scat1.set_label("Stochastic Gradient Descent")
    scat2.set_offsets(([momentum_X[n], momentum_X[n] ** 4 - 50 * momentum_X[n] ** 3 - momentum_X[n] + 1]))
    scat2.set_label("Gradien Descent with Momentum")
    plt.legend(bbox_to_anchor=(0.8, 1.0))
    print("Frams:%d" % n, sgd_X[n])
    return scat1, scat2

def momentum_update(n, fig, scat):
    scat.set_offsets(([momentum_X[n], momentum_X[n] ** 4 - 50 * momentum_X[n] ** 3 - momentum_X[n] + 1]))
    scat.set_label("Gradien Descent with Momentum")
    plt.legend(bbox_to_anchor=(0.8, 1.0))
    print("Frams:%d" % n, momentum_X[n])
    return scat

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# anim = animation.FuncAnimation(fig, func=update, frames=len(sgd_X), interval=10)
plt.plot(t, t ** 4 - 50 * (t ** 3) - t + 1)
plt.xlabel("data(x)")
plt.ylabel("Function X^4-50X^3-X+1")
anim = animation.FuncAnimation(fig=fig, func=sgd_update, fargs=(fig, sgd_PLOT, momentum_PLOT), frames=len(sgd_X), interval=50)
# momentum_anim = animation.FuncAnimation(fig=fig, func=momentum_update, fargs=(fig, momentum_PLOT), frames=len(momentum_X), interval=50)
# plt.show()



# anim.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
anim.save('anim.mp4', writer=writer)

