import math
import random
import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
print(keras.__version__)

keras.datasets.mnist.load_data(path="mnist.npz")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train / 255
x_test = x_test / 255

w_hidden1 = [[random.uniform(-0.1, 0.1) for _ in range(len(x_train[0]))] for _ in range(128)]
w_hidden2 = [[random.uniform(-0.1, 0.1) for _ in range(129)] for _ in range(64)]
w_output = [[random.uniform(-0.1, 0.1) for _ in range(65)] for _ in range(10)]



def ReLU(x):
    return max(0, x)

def ableitung_von_ReLU(x):
    return 1 if x > 0 else 0

def softmax(z_out):
    max_z = max(z_out)
    exp_values = [math.exp(z - max_z) for z in z_out]
    sum_exp = sum(exp_values)
    return [v/sum_exp for v in exp_values] 

def loss_funktion(y_vorhersage, y_batch):
    loss = 0
    loss += (y_batch[i][j] * math.log(y_vorhersage[j] + 1e-9) for j in range(10))
    return -loss

def y_transformieren(y):
    liste = [0.0]*10
    liste[y] = 1
    return liste

y_train = [y_transformieren(y) for y in y_train]
y_test = [y_transformieren(y) for y in y_test]

            

def pred():
    i = 0
    richtig = 0
    falsch = 1
    for i in range(len(x_test)):
        z1_hidden = [sum(w_hidden1[h][j] * x_test[i][j] for j in range(len(x_test[0]))) for h in range(128)]
        a1 = [ReLU(z1_hidden[j]) for j in range(128)] + [1.0]
        z2_hidden = [sum(a1[j]*w_hidden2[h][j] for j in range(129)) for h in range(64)]
        a2 = [ReLU(z2_hidden[j]) for j in range(64)] + [1.0]
        z_out = [sum(a2[j]*w_output[h][j] for j in range(65)) for h in range(10)]
        y_vorhersage = softmax(z_out)
        y_max_vhs = np.argmax(y_vorhersage)
        y_max_test = np.argmax(y_test[i])

        if y_max_vhs == y_max_test:
            richtig += 1

    print(richtig / len(x_test))

l_xtrain0 = len(x_train[0])

learning_rate = 0.01
for epoch in range(1):
    print(epoch)
    batch_size = 64

    for start in range(0, len(x_train), batch_size):
        end = start + batch_size  
        x_batch = x_train[start:end]
        y_batch = y_train[start:end]
        d_gradient_w_hidden1 = [[0 for _ in range(l_xtrain0)] for _ in range(128)]
        d_gradient_w_hidden2 = [[0 for _ in range(129)] for _ in range(64)]
        d_gradient_w_out = [[0 for _ in range(65)] for _ in range(10)]
        print(end)
        l_xbatch0 = len(x_batch[0])
        

        for i in range(len(x_batch)):

            # Forward Pass
            z1_hidden = [sum(w_hidden1[h][j] * x_batch[i][j] for j in range(l_xtrain0)) for h in range(128)]
            a1 = [ReLU(z1_hidden[j]) for j in range(128)] + [1.0]
            z2_hidden = [sum(a1[j]*w_hidden2[h][j] for j in range(129)) for h in range(64)]
            a2 = [ReLU(z2_hidden[j]) for j in range(64)] + [1.0]
            z_out = [sum(a2[j]*w_output[h][j] for j in range(65)) for h in range(10)]
            y_vorhersage = softmax(z_out)

            #loss_funktion(y_vorhersage, y_batch)
            
            # Backpropagation
            delta_out = [y_vorhersage[j] - y_batch[i][j] for j in range(10)]
            gradient_w_out = [[delta_out[h] * a2[j] for j in range(65)] for h in range(10)]

            delta_hidden2 = [sum(delta_out[h] * w_output[h][j]  for h in range(10)) * ableitung_von_ReLU(z2_hidden[j]) for j in range(64)]
            gradient_w_hidden2 = [[delta_hidden2[h] * a1[j] for j in range(129)] for h in range(64)]
            
            delta_hidden1 = [sum(delta_hidden2[h] * w_hidden2[h][j]  for h in range(64)) * ableitung_von_ReLU(z1_hidden[j]) for j in range(128)]
            gradient_w_hidden1 = [[delta_hidden1[h] * x_batch[i][j] for j in range(l_xbatch0)] for h in range(128)]

            # addieren zum Durchschnittsgradienten
            for j in range(128):
                for h in range(l_xbatch0):
                    d_gradient_w_hidden1[j][h] += gradient_w_hidden1[j][h]
            for j in range(64):
                for h in range(129):
                    d_gradient_w_hidden2[j][h] += gradient_w_hidden2[j][h]
            for j in range(10):
                for h in range(65):
                    d_gradient_w_out[j][h] += gradient_w_out[j][h]
        
        # berechnen des Durchsschnittsgradienten
        for j in range(128):
            for h in range(l_xbatch0):
                d_gradient_w_hidden1[j][h] /= len(x_batch)
        for j in range(64):
            for h in range(129):
                d_gradient_w_hidden2[j][h] /= len(x_batch)
        for j in range(10):
            for h in range(65):
                d_gradient_w_out[j][h] /= len(x_batch)

        # Gewichtsupdate
        for j in range(128):
            for h in range(l_xbatch0):
                w_hidden1[j][h] -= learning_rate * d_gradient_w_hidden1[j][h]
        for j in range(64):
            for h in range(129):
                w_hidden2[j][h] -= learning_rate * d_gradient_w_hidden2[j][h]
        for j in range(10):
            for h in range(65):
                w_output[j][h] -= learning_rate * d_gradient_w_out[j][h]
        

    pred()