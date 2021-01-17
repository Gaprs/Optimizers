# Gradient-Desctent-by-Animation
**********GIF by Optimizers result**********

![video](https://github.com/Gaprs/Gradient-Desctent-by-Animation/blob/master/Optimizers.gif)

Optimizers function use x^4 - 50*x^3 - x + 1
The function differential 3*x^3 - 150*x^2 - 1

Optimizers function include SGD, Momentum, Nesterov, Adagrade, RMSprop, Adam

SGD:  w' = w - lr * (partial(L) / partial(w)), L is Loss function, w is weight, lr is learning rate.

Momentum_version1: 

                   v' = beta * v - lr * (partial(L) / partial(w))
                   w' = w + v'
                   
Momentum_version2: 

                   v' = beta * v + (1-beta)*(partial(L) / partial(w))
                   w' = w + lr * v'

Nesterov: 

          v' = beta * v - lr * (partial(L) / partial(w))
          w' = w + v'
          v* = beta * v - lr * (partial(L) / partial(w'))
          w* = w + v*

Adagrade: 

![image](https://github.com/Gaprs/Gradient-Desctent-by-Animation/blob/master/Adagrade.png)


RMSprop(Root Mean Square prop, 方均根傳播):

![image](https://github.com/Gaprs/Gradient-Desctent-by-Animation/blob/master/RMSprop.JPG)


Adam:

![image](https://github.com/Gaprs/Gradient-Desctent-by-Animation/blob/master/Adam.png)


