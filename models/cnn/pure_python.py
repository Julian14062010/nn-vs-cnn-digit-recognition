import math
import random
import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
print(keras.__version__)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

class ConvolutionNeuralNetwork():
    
    def __init__(self):
        super().__init__()

        self.x_train = x_train.reshape(60000, 28, 28)
        self.x_test = x_test.reshape(10000, 28, 28)
        self.y_train = [self.y_transformieren(y) for y in y_train]
        self.y_test = [self.y_transformieren(y) for y in y_test]

        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255

        # -------- Parameter --------

        self.num1_filter = 32
        self.filter_size1 = 3
        self.conv1_out_size = 26
        self.num2_filter = 64
        self.filter_size2 = 3
        self.pool1_out = self.conv1_out_size // 2
        self.conv2_out_size = self.pool1_out - self.filter_size2 + 1
        self.pooled = (self.conv2_out_size - self.conv2_out_size % 2) // 2
        self.hidden_neurons = 128
        self.output_neurons = 10

        # Gewichte
        self.w_hidden = [[random.uniform(-0.1, 0.1) for _ in range(self.num2_filter * self.pooled * self.pooled)]for _ in range(self.hidden_neurons)]
        self.w_output = [[random.uniform(-0.1, 0.1) for _ in range(self.hidden_neurons+1)]for _ in range(self.output_neurons)]

        self.filter1 = [[[random.uniform(-0.1, 0.1) for _ in range(self.filter_size1)]for _ in range(self.filter_size1)]for _ in range(self.num1_filter)]
        self.filter2 = [[[[random.uniform(-0.1, 0.1) for _ in range(self.filter_size2)] for _ in range(self.filter_size2)] for _ in range(self.num1_filter)] for _ in range(self.num2_filter)]
        self.bias1_filter = [random.uniform(-0.1, 0.1) for _ in range(self.num1_filter)]
        self.bias2_filter = [random.uniform(-0.1, 0.1) for _ in range(self.num2_filter)]


        self.lr = 0.001
        self.epochs = 10
        self.batch_size = 64

    def y_transformieren(self, y):
        liste = [0.0]*10
        liste[y] = 1
        return liste


    def ReLU(self, x):
        return max(0, x)

    def ableitung_von_ReLU(self, x):
        return 1 if x > 0 else 0

    def softmax(self, z_out):
        max_z = max(z_out)
        exp_values = [math.exp(z - max_z) for z in z_out]
        sum_exp = sum(exp_values)
        return [v/sum_exp for v in exp_values] 



    

    def fit(self):


        for epoch in range(self.epochs):
            self.batch_size = 64

            for start in range(0, len(self.x_train), self.batch_size):
                end = start + self.batch_size
                x_batch = self.x_train[start:end]
                y_batch = self.y_train[start:end]

                
                d_gradient_w_hidden = [[0 for _ in range(self.num2_filter * self.pooled * self.pooled)] for _ in range(self.hidden_neurons)]
                d_gradient_w_out = [[0 for _ in range(self.hidden_neurons+1)] for _ in range(self.output_neurons)]
                d_gradient_filter1 = [[[0 for _ in range(self.filter_size1)] for _ in range(self.filter_size1)] for _ in range(self.num1_filter)]
                d_gradient_bias1 = [0 for _ in range(self.num1_filter)]
                d_gradient_filter2 = [[[[0 for _ in range(self.filter_size2)] for _ in range(self.filter_size2)] for _ in range(self.num1_filter)] for _ in range(self.num2_filter)]
                d_gradient_bias2 = [0 for _ in range(self.num2_filter)]
                
                
                for i in range(len(x_batch)):
                    print(i)
                    
                    # -------- Forward Pass --------

                    # Filter
                    Z = [[[0 for _ in range(self.conv1_out_size)] for _ in range(self.conv1_out_size)] for _ in range(self.num1_filter)]
                    A = [[[0 for _ in range(self.conv1_out_size)] for _ in range(self.conv1_out_size)] for _ in range(self.num1_filter)]
                    Z2 = [[[0 for _ in range(self.conv2_out_size)] for _ in range(self.conv2_out_size)] for _ in range(self.num2_filter)]
                    A2 = [[[0 for _ in range(self.conv2_out_size)] for _ in range(self.conv2_out_size)] for _ in range(self.num2_filter)]
                    mask1 = [[[[0,0,0,0] for _ in range(self.conv1_out_size // 2)] for _ in range(self.conv1_out_size // 2)] for _ in range(self.num1_filter)]  
                    mask2 = [[[[0,0,0,0] for _ in range(self.pooled)] for _ in range(self.pooled)] for _ in range(self.num2_filter)]  

                    # Convolution 1
                    for j in range(self.num1_filter):
                        for n in range(self.conv1_out_size):
                            for m in range(self.conv1_out_size):
                                s = 0
                                for h in range(self.filter_size1):
                                    for w in range(self.filter_size1):
                                        s += self.filter1[j][h][w] * x_batch[i][n+h][m+w]
                                Z[j][n][m] = s + self.bias1_filter[j]
                                A[j][n][m] = self.ReLU(Z[j][n][m])

                                
                    # Pooling 1            
                    P1 = [[[0 for _ in range(self.conv1_out_size // 2)] for _ in range(self.conv1_out_size // 2)] for _ in range(self.num1_filter)]
                    for j in range(self.num1_filter):
                        for n in range(self.conv1_out_size // 2):
                            for m in range(self.conv1_out_size // 2):
                                window1 = [
                                    A[j][2*n][2*m],
                                    A[j][2*n+1][2*m],
                                    A[j][2*n][2*m+1],
                                    A[j][2*n+1][2*m+1]
                                ]
                                idx_max1 = max(range(4), key=lambda idx: window1[idx])
                                mask1[j][n][m][idx_max1] = 1
                                P1[j][n][m] = window1[idx_max1]

                    
                    

                    # Convolution 2
                    for j in range(self.num2_filter):
                        for n in range(self.conv2_out_size):
                            for m in range(self.conv2_out_size):
                                s = 0
                                for k in range(self.num1_filter):
                                    for h in range(self.filter_size2):
                                        for w in range(self.filter_size2):
                                            s += self.filter2[j][k][h][w] * P1[k][n+h][m+w]
                                Z2[j][n][m] = s + self.bias2_filter[j]
                                A2[j][n][m] = self.ReLU(Z2[j][n][m])

                    
                    P2 = [[[0 for _ in range(self.pooled)] for _ in range(self.pooled)] for _ in range(self.num2_filter)]
                    for j in range(self.num2_filter):
                        for n in range(self.pooled):
                            for m in range(self.pooled):
                                window2 = [
                                    A2[j][2*n][2*m],
                                    A2[j][2*n+1][2*m],
                                    A2[j][2*n][2*m+1],
                                    A2[j][2*n+1][2*m+1]
                                ]
                                idx_max2 = max(range(4), key=lambda idx: window2[idx])
                                mask2[j][n][m][idx_max2] = 1
                                P2[j][n][m] = window2[idx_max2]


                    # Flatten
                    x = []
                    for j in range(self.num2_filter):
                        for n in range(self.pooled):
                            for m in range(self.pooled):
                                x.append(P2[j][n][m])

                    # Hidden Layer 1
                    z_hidden = []
                    for j in range(self.hidden_neurons):
                        s = sum(x[h] * self.w_hidden[j][h] for h in range(len(x)))
                        z_hidden.append(s)
                    a = [self.ReLU(z_hidden[j]) for j in range(self.hidden_neurons)] + [1.0]  

                    # Output Layer
                    z_out = []
                    for j in range(self.output_neurons):
                        s = sum(a[h] * self.w_output[j][h] for h in range(len(a)))
                        z_out.append(s)
                    y_pred = self.softmax(z_out)

                    # -------- Backpropagation --------

                    # Output Gradient
                    delta_out = [y_pred[j] - y_batch[i][j] for j in range(self.output_neurons)]
                    gradient_w_out = [[delta_out[j] * a[h] for h in range(len(a))] for j in range(self.output_neurons)]

                    # Hidden Layer 1 Gradient
                    delta_hidden = []
                    for j in range(self.hidden_neurons):
                        s = sum(delta_out[h] * self.w_output[h][j] for h in range(self.output_neurons))
                        delta_hidden.append(s * self.ableitung_von_ReLU(z_hidden[j]))
                    self.gradient_w_hidden = [[delta_hidden[j] * x[h] for h in range(len(x))] for j in range(self.hidden_neurons)]

                    delta_pool_reshaped = [0 for _ in range(self.num2_filter*(self.pooled)*(self.pooled))]
                    for h in range(len(x)):
                        s = 0
                        for j in range(self.hidden_neurons):
                            s += delta_hidden[j] * self.w_hidden[j][h]
                        delta_pool_reshaped[h] = s

                
                    delta_pool2 = [[[0 for _ in range(self.pooled)] for _ in range(self.pooled)] for _ in range(self.num2_filter)]
                    idx = 0
                    for j in range(self.num2_filter):
                        for n in range(self.pooled):
                            for m in range(self.pooled):
                                delta_pool2[j][n][m] = delta_pool_reshaped[idx]
                                idx += 1

                
                    delta_A2 = [[[0 for _ in range(self.conv2_out_size)] for _ in range(self.conv2_out_size)] for _ in range(self.num2_filter)]
                    for j in range(self.num2_filter):
                        for n in range(self.pooled):
                            for m in range(self.pooled):
                                if mask2[j][n][m][0]:
                                    delta_A2[j][2*n][2*m] = delta_pool2[j][n][m]
                                if mask2[j][n][m][1]:
                                    delta_A2[j][2*n+1][2*m] = delta_pool2[j][n][m]
                                if mask2[j][n][m][2]:
                                    delta_A2[j][2*n][2*m+1] = delta_pool2[j][n][m]
                                if mask2[j][n][m][3]:
                                    delta_A2[j][2*n+1][2*m+1] = delta_pool2[j][n][m]

                    
                    
                    delta_z2 = [[[delta_A2[j][n][m] * self.ableitung_von_ReLU(Z2[j][n][m]) for m in range(self.conv2_out_size)] for n in range(self.conv2_out_size)] for j in range(self.num2_filter)]
                    
                    # Gradient Filter 2
                    gradient_filter2 = [[[[0 for _ in range(self.filter_size2)] for _ in range(self.filter_size2)] for _ in range(self.num1_filter)] for _ in range(self.num2_filter)]
                    for j in range(self.num2_filter):
                        for k in range(self.num1_filter):
                            for h in range(self.filter_size2):
                                for w in range(self.filter_size2):
                                    s = 0
                                    for n in range(self.conv2_out_size):
                                        for m in range(self.conv2_out_size):
                                            s += delta_z2[j][n][m] * P1[k][n+h][m+w]
                                    gradient_filter2[j][k][h][w] = s
                    gradient_bias2 = [sum(delta_z2[j][n][m] for n in range(self.conv2_out_size) for m in range(self.conv2_out_size)) for j in range(self.num2_filter)]

                    # Filter rotieren
                    self.filter2_rotiert = [[[[0 for _ in range(self.filter_size2)] for _ in range(self.filter_size2)] for _ in range(self.num1_filter)] for _ in range(self.num2_filter)]
                    for j in range(self.num2_filter):
                        for k in range(self.num1_filter):
                            for n in range(self.filter_size2):
                                for m in range(self.filter_size2):
                                    self.filter2_rotiert[j][k][n][m] = self.filter2[j][k][self.filter_size2- 1 - n][self.filter_size2 - 1 - m]

                    # Delta A1
                    delta_pool1 = [[[0 for _ in range(self.conv1_out_size)] for _ in range(self.conv1_out_size)] for _ in range(self.num1_filter)]
                    for j in range(self.num1_filter):
                        for h in range(self.conv1_out_size):
                            for w in range(self.conv1_out_size):
                                s = 0
                                for k in range(self.num2_filter):
                                    for n in range(self.filter_size2):
                                        for m in range(self.filter_size2):
                                            ih = h + n - (self.filter_size2 - 1)
                                            iw = w + m - (self.filter_size2 - 1)

                                            if 0 <= ih < self.conv2_out_size and 0 <= iw < self.conv2_out_size:
                                                s += delta_z2[k][ih][iw] * self.filter2_rotiert[k][j][n][m]
                                delta_pool1[j][h][w] = s     


                    delta_A1 = [[[0 for _ in range(self.conv1_out_size)] for _ in range(self.conv1_out_size)] for _ in range(self.num1_filter)]
                    for j in range(self.num1_filter):
                        for n in range(self.conv1_out_size // 2):
                            for m in range(self.conv1_out_size // 2):
                                if mask1[j][n][m][0]:
                                    delta_A1[j][2*n][2*m] = delta_pool1[j][n][m]
                                if mask1[j][n][m][1]:
                                    delta_A1[j][2*n+1][2*m] = delta_pool1[j][n][m]
                                if mask1[j][n][m][2]:
                                    delta_A1[j][2*n][2*m+1] = delta_pool1[j][n][m]
                                if mask1[j][n][m][3]:
                                    delta_A1[j][2*n+1][2*m+1] = delta_pool1[j][n][m]


                    # Gradient Filter 1
                    delta_z1 = [[[delta_A1[j][n][m] * self.ableitung_von_ReLU(Z[j][n][m]) for m in range(self.conv1_out_size)] for n in range(self.conv1_out_size)] for j in range(self.num1_filter)]
                    gradient_filter1 = [[[0 for _ in range(self.filter_size1)] for _ in range(self.filter_size1)] for _ in range(self.num1_filter)]
                    for j in range(self.num1_filter):
                        for h in range(self.filter_size1):
                            for w in range(self.filter_size1):
                                s = 0
                                for n in range(self.conv1_out_size):
                                    for m in range(self.conv1_out_size):
                                        s += delta_z1[j][n][m] * x_batch[i][n+h][m+w]
                                gradient_filter1[j][h][w] = s
                    gradient_bias1 = [sum(delta_z1[j][n][m] for n in range(self.conv1_out_size) for m in range(self.conv1_out_size)) for j in range(self.num1_filter)]
                    
                    # Summe aller Gradienten
                    for j in range(self.hidden_neurons):
                        for h in range(self.num2_filter*(self.pooled)*(self.pooled)):
                            d_gradient_w_hidden[j][h] += self.gradient_w_hidden[j][h]
                    for j in range(self.output_neurons):
                        for h in range(self.hidden_neurons+1):
                            d_gradient_w_out[j][h] += gradient_w_out[j][h]
                    for j in range(self.num2_filter):
                        for k in range(self.num1_filter):
                            for n in range(self.filter_size2):
                                for m in range(self.filter_size2):
                                    d_gradient_filter2[j][k][n][m] += gradient_filter2[j][k][n][m]
                        d_gradient_bias2[j] += gradient_bias2[j]
                    for j in range(self.num1_filter):
                        for h in range(self.filter_size1):
                            for w in range(self.filter_size1):
                                d_gradient_filter1[j][h][w] += gradient_filter1[j][h][w]
                        d_gradient_bias1[j] += gradient_bias1[j]

                # Durchschnittsgradienten
                batch_len = len(x_batch)
                for j in range(self.hidden_neurons):
                    for h in range(self.num2_filter*(self.pooled)*(self.pooled)):
                        d_gradient_w_hidden[j][h] /= batch_len
                for j in range(self.output_neurons):
                    for h in range(self.hidden_neurons+1):
                        d_gradient_w_out[j][h] /= batch_len
                for j in range(self.num2_filter):
                        for k in range(self.num1_filter):
                            for n in range(self.filter_size2):
                                for m in range(self.filter_size2):
                                    d_gradient_filter2[j][k][n][m] /= batch_len
                        d_gradient_bias2[j] /= batch_len
                for j in range(self.num1_filter):
                    for h in range(self.filter_size1):
                        for w in range(self.filter_size1):
                            d_gradient_filter1[j][h][w] /= batch_len
                    d_gradient_bias1[j] /= batch_len

                # Gewichtsupdate
                for j in range(self.hidden_neurons):
                    for h in range(self.num2_filter*(self.pooled)*(self.pooled)):
                        self.w_hidden[j][h] -= self.lr * d_gradient_w_hidden[j][h]
                for j in range(self.output_neurons):
                    for h in range(self.hidden_neurons+1):
                        self.w_output[j][h] -= self.lr * d_gradient_w_out[j][h]
                for j in range(self.num2_filter):
                        for k in range(self.num1_filter):
                            for n in range(self.filter_size2):
                                for m in range(self.filter_size2):
                                    self.filter2[j][k][n][m] -= self.lr * d_gradient_filter2[j][k][n][m]
                        self.bias2_filter[j] -= self.lr * d_gradient_bias2[j]
                for j in range(self.num1_filter):
                    for h in range(self.filter_size1):
                        for w in range(self.filter_size1):
                            self.filter1[j][h][w] -= self.lr * d_gradient_filter1[j][h][w]
                    self.bias1_filter[j] -= self.lr * d_gradient_bias1[j]


    def pred(self):
        richtig = 0
        counts = [0]*10

        for i in range(len(self.x_test)):

            Z = [[[self.ReLU(
                sum(self.filter1[j][h][w] * self.x_test[i][n+h][m+w]
                    for h in range(self.filter_size1)
                    for w in range(self.filter_size1)
                ) + self.bias1_filter[j])
                for m in range(self.conv1_out_size)]
                for n in range(self.conv1_out_size)]
                for j in range(self.num1_filter)]

            P1 = [[[max(
                Z[j][2*n][2*m],
                Z[j][2*n+1][2*m],
                Z[j][2*n][2*m+1],
                Z[j][2*n+1][2*m+1]
            ) for m in range(self.pool1_out)]
                for n in range(self.pool1_out)]
                for j in range(self.num1_filter)]   

            Z2 = [[[self.ReLU(
                sum(self.filter2[j][k][fh][fw] * P1[k][h+fh][w+fw]
                    for k in range(self.num1_filter)
                    for fh in range(self.filter_size2)
                    for fw in range(self.filter_size2)
                ) + self.bias2_filter[j])
                for w in range(self.conv2_out_size)]
                for h in range(self.conv2_out_size)]
                for j in range(self.num2_filter)]

            P2 = [[[max(
                Z2[j][2*n][2*m],
                Z2[j][2*n+1][2*m],
                Z2[j][2*n][2*m+1],
                Z2[j][2*n+1][2*m+1]
            ) for m in range(self.pooled)]
                for n in range(self.pooled)]
                for j in range(self.num2_filter)]

            x = []
            for j in range(self.num2_filter):
                for n in range(self.pooled):
                    for m in range(self.pooled):
                        x.append(P2[j][n][m])

            z_hidden = [sum(x[h] * self.w_hidden[j][h] for h in range(len(x)))
                        for j in range(self.hidden_neurons)]

            a = [self.ReLU(z_hidden[j]) for j in range(self.hidden_neurons)] + [1.0]

            z_out = [sum(a[h] * self.w_output[j][h] for h in range(len(a)))
                    for j in range(self.output_neurons)]

            y_vorhersage = self.softmax(z_out)

            y_max_vhs = max(range(self.output_neurons), key=lambda k: y_vorhersage[k])
            y = max(range(self.output_neurons), key=lambda k: self.y_test[i][k])

            counts[y_max_vhs] += 1

            if y_max_vhs == y:
                richtig += 1

        print(counts)
        print("Accuracy:", richtig / len(self.x_test))



cnn = ConvolutionNeuralNetwork()
cnn.fit()
cnn.pred()