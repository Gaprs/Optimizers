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


Adam:(momentum+RMSprop)

momentum :

        Vdw = beta1 * Vdw + (1-beta1) * dw, Vdb = beta1 * Vdb + (1-beta1) * db;
        (Corrected) Vdw = Vdw/(1-beta1^t), Vdb = Vdb/(1-beta1^t)
        beta1是用來是用來計算dw的移動加權平均(動量)

RMSprop:

        Sdw = beta2 * Sdw + (1-beta2) * dw^2, Sbd = beta2 * Sdb + (1-beta2) * db^2;
        (Corrected) Sdw = Sdw/(1-beta2^t), Sdb = Sdb/(1-beta2^t)
        beta2是用來是用來計算dw^2 and db^2的移動加權平均

update:

        W = W - lr * ((Corrected Vdw) / (Corrected Sdw)^1/2 + epsilon
        b = b - lr * ((Corrected Vdw) / (Corrected Sdb)^1/2 + epsilon


![image](https://github.com/Gaprs/Gradient-Desctent-by-Animation/blob/master/Adam.png)


