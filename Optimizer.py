import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from matplotlib import rcParams

fig, ax = plt.subplots()

lr = 0.0001 # learning rate
t = np.arange(-30, 60, 0.1) #compute minimum output -3 ~ 3
x = -25 # 起始值x
y = x ** 4 - 50 * x ** 3 - x + 1
dy_dx = 4 * x ** 3 - 150 * x ** 2 - 1 # Gradient, 對權重x微分
epoch = 150 # train times 120
min = [] # a list, for compute minimum output y

sgd_x = 0 # initial update weight x by SGD
sgd_dy_dx = 0
nes_dy_dx = 0
sgdX = []
sgdY = []

momentum_x = 0
momentum_v = 0
nesterov_x = 0
nesterov_v = 0
v = 0
beta = 0.9
momentumX = []
momentumY = []
nesterovX = []
nesterovY = []

epsilon = 1e-7
Grad = 0
GradX = []
GradY = []

Rho = 0.9
RMS_Grad = 0
RMSX = []
RMSY = []

beta1 = 0.9
beta2 = 0.999
epsilon_Adam = 1e-8
Adam_Grad = 0
AdamX = []
AdamY = []
Adam_m = 0
Adam_v = 0
Adam_updateM = 0
Adam_updateV = 0

plt.plot(t, t**4 - 50*t**3 - t + 1)


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

def Momentum(x, v, beta, dy_dx, momentum_x, lr):
    if momentum_x == 0:
        v = beta * v - lr * dy_dx
        x = x + v
        return x, v
    else:
        v = beta * v - lr * dy_dx
        momentum_x = momentum_x + v
        return momentum_x, v
def Nesterov(x, v, beta, dy_dx, nesterov_x, nesterov_v, lr):
    if nesterov_x == 0:
        v_Nes = beta * v - lr * dy_dx
        x_Nes = x + v_Nes
        nes_dy_dx = 4 * x_Nes ** 3 - 150 * x_Nes ** 2 - 1
        v = beta * v - lr * nes_dy_dx
        x = x + v
        return x, v
    else:
        v_Nes = beta * v - lr * dy_dx
        x_Nes = nesterov_x + v_Nes
        nes_dy_dx = 4 * x_Nes ** 3 - 150 * x_Nes ** 2 - 1
        nesterov_v = beta * nesterov_v - lr * nes_dy_dx
        nesterov_x = nesterov_x + v
        return nesterov_x, nesterov_v

def Adagrade(x , epsilon, dy_dx, Grad, lr):
    Grad = Grad + (dy_dx*dy_dx)
    lr = 10
    x = x - (lr*dy_dx) / ((Grad**0.5)+epsilon)
    return Grad, x

def RMSprop(x, epsilon, dy_dx, RMS_Grad, Rho, lr, epoch=1):
    lr = 1
    RMS_Grad = (Rho*(RMS_Grad+(dy_dx*dy_dx))/epoch) + (1-Rho) * (dy_dx**2)
    x = x -(lr * dy_dx) / ((RMS_Grad**0.5)+epsilon)
    return RMS_Grad, x

def Adam(x, epsilon, Adam_m, Adam_v, dy_dx, Adam_Grad, lr):
    lr = 1
    Adam_m = beta1 * Adam_m + (1 - beta1) * dy_dx
    Adam_v = beta2 * Adam_v + (1 - beta2) * (dy_dx**2)
    Adam_updateM = Adam_m / (1 - beta1)
    Adam_updateV = Adam_v / (1 - beta2)
    x = x - (lr * Adam_updateM) / ((Adam_updateV**0.5)+epsilon_Adam)
    return Adam_m, Adam_v, x

def Adam_plot(x, epsilon, Adam_m, Adam_v, dy_dx, Adam_Grad, lr):
    for i in range(epoch):
        dy_dx = 4 * x ** 3 - 150 * x ** 2 - 1
        Adam_m, Adam_v, x = Adam(x, epsilon, Adam_m, Adam_v, dy_dx, Adam_Grad, lr)
        y = x ** 4 - 50 * x ** 3 -x + 1
        z = np.abs(y - min_y)
        if z < 0.01:
            break
        if i % 1 == 0:
            AdamX.append(x)
            AdamY.append(y)
    Adam_plot = plt.scatter(AdamX, AdamY, marker='o', color='', edgecolors='Y')
    return AdamX, Adam_plot

def RMSprop_plot(x, epsilon, dy_dx, RMS_Grad, lr):
    for i in range(epoch):
        if i == 0:
            RMS_Grad, x = RMSprop(x, epsilon, dy_dx, RMS_Grad, Rho, lr)
        else:
            dy_dx = 4 * x ** 3 - 150 * x ** 2 - 1
            RMS_Grad, x = RMSprop(x, epsilon, dy_dx, RMS_Grad, Rho, lr, i+1)
        y = x ** 4 - 50 * x ** 3 -x + 1
        z = np.abs(y - min_y)
        if z < 0.01:
            break
        if i % 1 == 0:
            RMSX.append(x)
            RMSY.append(y)
    RMS_plot = plt.scatter(RMSX, RMSY, marker='o', color='', edgecolors='black')
    return RMSX, RMS_plot

def Adagrade_plot(x , epsilon, dy_dx, Grad, lr):
    for i in range(epoch):
        if i == 0:
            Grad, x = Adagrade(x, epsilon, dy_dx, Grad, lr)
        else:
            dy_dx = 4 * x ** 3 - 150 * x ** 2 - 1
            Grad, x = Adagrade(x, epsilon, dy_dx, Grad, lr)
        y = x ** 4 - 50 * x ** 3 - x + 1
        z = np.abs(y - min_y)
        if z < 0.01:
            break
        if i % 1 == 0:
            GradX.append(x)
            GradY.append(y)
    Adagrade_plot = plt.scatter(GradX, GradY, marker='o', color='', edgecolors='brown')
    return GradX, Adagrade_plot

def Nesterov_plot(x, v, dy_dx, nesterov_x, nesterov_v, min_y):
    for i in range(epoch):
        if i == 0:
            nesterov_x, nesterov_v = Nesterov(x, v, beta, dy_dx, nesterov_x, nesterov_v, lr)
        else:
            nes_dy_dx = 4 * nesterov_x ** 3 - 150 * nesterov_x ** 2 - 1
            nesterov_x, nesterov_v = Nesterov(x, nesterov_v, beta, nes_dy_dx, nesterov_x, nesterov_v, lr)
        y = nesterov_x ** 4 - 50 * nesterov_x ** 3 - nesterov_x + 1
        z = np.abs(y - min_y)
        if z < 0.01:
            break
        if i % 1 == 0:
            nesterovX.append(nesterov_x)
            nesterovY.append(y)
    Nesterov_plot = plt.scatter(nesterovX, nesterovY, marker='o', color='', edgecolors='b')
    return nesterovX, Nesterov_plot

def sgd_plot(x, sgd_x, min_y):
    for i in range(epoch):
        if i == 0:
            sgd_x = SGD(x, lr, dy_dx, sgd_x)
        else:
            sgd_dy_dx = 4 * sgd_x ** 3 - 150 * sgd_x ** 2 - 1
            sgd_x = SGD(x, lr, sgd_dy_dx, sgd_x)
        y = sgd_x ** 4 - 50 * sgd_x ** 3 - sgd_x + 1
        z = np.abs(y - min_y)
        if z < 0.01:
            break
        if i % 1 == 0:
            sgdX.append(sgd_x)
            sgdY.append(y)
    sgd_plot = plt.scatter(sgdX, sgdY, marker='o', color='', edgecolors='r')
    return sgdX, sgd_plot

def momentum_plot(x, v, momentum_x, momentum_v, min_y):
    for i in range(epoch):
        if i == 0:
            momentum_x, momentum_v = Momentum(x, v, beta, dy_dx, momentum_x, lr)
        else:
            momentum_dy_dx = 4 * momentum_x ** 3 - 150 * momentum_x ** 2 - 1
            momentum_x, momentum_v = Momentum(x, momentum_v, beta, momentum_dy_dx, momentum_x, lr)
        y = momentum_x ** 4 - 50 * momentum_x ** 3 - momentum_x + 1
        z = np.abs(y - min_y)
        if z < 0.01:
            break
        if i % 1 == 0:
            momentumX.append(momentum_x)
            momentumY.append(y)
    momentum_plot = plt.scatter(momentumX, momentumY, marker='o', color='', edgecolors='g')
    return momentumX, momentum_plot

def update(n, fig, scat1, scat2, scat3, scat4, scat5, scat6):

    scat1.set_offsets(([sgd_X[n], sgd_X[n] ** 4 - 50 * sgd_X[n] ** 3 - sgd_X[n] + 1]))
    scat1.set_label("Gradient Descent")
    scat2.set_offsets(([momentum_X[n], momentum_X[n] ** 4 - 50 * momentum_X[n] ** 3 - momentum_X[n] + 1]))
    scat2.set_label("Gradient Descent with Momentum")
    scat3.set_offsets(([nesterov_X[n], nesterov_X[n] ** 4 - 50 * nesterov_X[n] ** 3 - nesterov_X[n] + 1]))
    scat3.set_label("Gradient Descent with Momentum & Nesterov")
    scat4.set_offsets(([Adagrade_X[n], Adagrade_X[n] ** 4 - 50 * Adagrade_X[n] ** 3 - Adagrade_X[n] + 1]))
    scat4.set_label("Gradient Descent with Adagrade")
    scat5.set_offsets(([RMSprop_X[n], RMSprop_X[n] ** 4 - 50 * RMSprop_X[n] ** 3 - RMSprop_X[n] + 1]))
    scat5.set_label("Gradient Descent with RMSprop")
    scat6.set_offsets(([Adam_X[n], Adam_X[n] ** 4 - 50 * Adam_X[n] ** 3 - Adam_X[n] + 1]))
    scat6.set_label("Gradient Descent with Adam")

    plt.legend(bbox_to_anchor=(0.5, 0.628))
    print("Frams:%d" %n, Adagrade_X[n])
    return scat1, scat2, scat3, scat4, scat5, scat6
# def update(n, fig, scat1, scat2):
#     scat1.set_offsets(([sgd_X[n], sgd_X[n] ** 4 - 50 * sgd_X[n] ** 3 - sgd_X[n] + 1]))
#     scat1.set_label("Stochastic Gradient Descent")
#     scat2.set_offsets(([momentum_X[n], momentum_X[n] ** 4 - 50 * momentum_X[n] ** 3 - momentum_X[n] + 1]))
#     scat2.set_label("Gradien Descent with Momentum")
#     plt.legend(bbox_to_anchor=(0.8, 1.0))
#     print("Frams:%d" % n, sgd_X[n])
#     return scat1, scat2


min_y = minimum_y()
sgd_X, sgd_PLOT = sgd_plot(x, sgd_x, min_y)
momentum_X, momen_p = momentum_plot(x, v, momentum_x, momentum_v, min_y)
nesterov_X, nesteron_p = Nesterov_plot(x, v, dy_dx, nesterov_x, nesterov_v, min_y)
Adagrade_X, Adagrade_p = Adagrade_plot(x , epsilon, dy_dx, Grad, lr)
RMSprop_X, RMSprop_p = RMSprop_plot(x, epsilon, dy_dx, RMS_Grad, lr)
Adam_X, Adam_p = Adam_plot(x, epsilon, Adam_m, Adam_v, dy_dx, Adam_Grad, lr)

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# anim = animation.FuncAnimation(fig=fig, func=update, fargs=(fig, sgd_PLOT, momen_p), frames=len(sgd_X), interval=50)
anim = animation.FuncAnimation(fig=fig, func=update, fargs=(fig, sgd_PLOT, momen_p, nesteron_p, Adagrade_p, RMSprop_p, Adam_p), frames=len(sgd_X), interval=50)

# plt.show()
wirter = animation.PillowWriter(fps=25)
anim.save("D:\Pycharm\Python_Porject\Optimizers.gif", writer=wirter)
