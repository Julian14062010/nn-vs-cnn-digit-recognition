import math
import random
#import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
print(keras.__version__)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Parameter
lr = 0.01
epochs = 10

# ______________Functions______________

def update_weight(weight, gradient):
    for i in range(len(weight)):
        for j in range(len(weight[0])):
            weight[i][j] -= lr * gradient[i][j]
    return weight
def update_bias(bias, gradient):
    for i in range(len(bias)):
        bias[i] -= lr * gradient[i]
    return bias
def transpose(x):
    return [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]
def zeros1(rows):
    return [0.0 for _ in range(rows)]
def zeros2(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]
def zeros3(a, b, c):
    return [[[0.0 for _ in range(c)] for _ in range(b)] for _ in range(a)]
def zeros4(a, b, c, d):
    return [[[[0.0 for _ in range(d)] for _ in range(c)] for _ in range(b)] for _ in range(a)]
def one_hot(y, num_classes=10):
    out = []
    for label in y:
        v = [0.0] * num_classes
        v[label] = 1.0
        out.append(v)
    return out
def cross_entropy(pred, target):
    return -sum(t * math.log(p + 1e-9) for p, t in zip(pred, target))

# ______________Classes______________

# ReLU
class ReLU:
    @staticmethod
    def forward(x):
        return max(0, x)
    def backward(x):
        return 1 if x > 0 else 0

# Convolution2D
class Conv2D:

    def __init__(self, num_filter, filter_size, input_depth):
        self.num_filter = num_filter
        self.filter_size = filter_size
        self.input_depth = input_depth
        if input_depth == 1:
            self.filter = [[[random.uniform(-0.1, 0.1) for _ in range(self.filter_size)] for _ in range(self.filter_size)] for _ in range(self.num_filter)]
        else:
            self.filter = [[[[random.uniform(-0.1, 0.1) for _ in range(self.filter_size)] for _ in range(self.filter_size)] for _ in range(self.input_depth)] for _ in range(self.num_filter)]
        self.bias = [random.uniform(-0.1, 0.1) for _ in range(self.num_filter)]

    def forward(self, x):

        self.x = x
        self.out_size = len(self.x[0]) - self.filter_size + 1
        self.Z = zeros3(self.num_filter, self.out_size, self.out_size)
        self.A = zeros3(self.num_filter, self.out_size, self.out_size)
        if self.input_depth == 1:
            for j in range(self.num_filter):
                for n in range(self.out_size):
                    for m in range(self.out_size):
                        s = 0
                        for h in range(self.filter_size):
                            for w in range(self.filter_size):
                                s += self.filter[j][h][w] * self.x[0][n+h][m+w]
                        self.Z[j][n][m] = s + self.bias[j]
                        self.A[j][n][m] = ReLU.forward(self.Z[j][n][m])
        else:
            for j in range(self.num_filter):
                for n in range(self.out_size):
                    for m in range(self.out_size):
                        s = 0
                        for k in range(self.input_depth):
                            for h in range(self.filter_size):
                                for w in range(self.filter_size):
                                    s += (self.filter[j][k][h][w] * self.x[k][n+h][m+w])
                        self.Z[j][n][m] = s + self.bias[j]
                        self.A[j][n][m] = ReLU.forward(self.Z[j][n][m])

        return self.A

    def backward(self, delta_z):
        delta_prev = zeros3(self.input_depth,len(self.x[0]),len(self.x[0][0]))
        for i in range(len(self.Z)):
            for j in range(len(self.Z[0])):
                for k in range(len(self.Z[0][0])):
                    delta_z[i][j][k] *= ReLU.backward(self.Z[i][j][k])

        if self.input_depth == 1:
            gradient_filter = zeros3(self.num_filter, self.filter_size, self.filter_size)

            for j in range(self.num_filter):
                for h in range(self.filter_size):
                    for w in range(self.filter_size):
                        s = 0
                        for n in range(self.out_size):
                            for m in range(self.out_size):
                                s += (delta_z[j][n][m] * self.x[0][n+h][m+w])
                        gradient_filter[j][h][w] = s
            for j in range(self.num_filter):
                for n in range(self.out_size):
                    for m in range(self.out_size):
                        for h in range(self.filter_size):
                            for w in range(self.filter_size):
                                delta_prev[0][n+h][m+w] += delta_z[j][n][m] * self.filter[j][h][w]

            for i in range(len(gradient_filter)):
                for j in range(len(gradient_filter[0])):
                    for k in range(len(gradient_filter[0][0])):
                        self.filter[i][j][k] -= lr * gradient_filter[i][j][k]

            gradient_bias = [sum(delta_z[j][n][m] for n in range(self.out_size) for m in range(self.out_size)) for j in range(self.num_filter)]

        else:

            gradient_filter = zeros4(self.num_filter, self.input_depth, self.filter_size, self.filter_size)

            for j in range(self.num_filter):
                for k in range(self.input_depth):
                    for h in range(self.filter_size):
                        for w in range(self.filter_size):
                            s = 0
                            for n in range(self.out_size):
                                for m in range(self.out_size):
                                    s += (delta_z[j][n][m] * self.x[k][n+h][m+w])
                            gradient_filter[j][k][h][w] = s

            for j in range(self.num_filter):
                for k in range(self.input_depth):
                    for n in range(self.out_size):
                        for m in range(self.out_size):
                            for h in range(self.filter_size):
                                for w in range(self.filter_size):
                                    delta_prev[k][n+h][m+w] += delta_z[j][n][m] * self.filter[j][k][h][w]

            gradient_bias = [sum(delta_z[j][n][m] for n in range(self.out_size) for m in range(self.out_size)) for j in range(self.num_filter)]

            for i in range(len(gradient_filter)):
                for j in range(len(gradient_filter[0])):
                    for k in range(len(gradient_filter[0][0])):
                        for n in range(len(gradient_filter[0][0][0])):
                            self.filter[i][j][k][n] -= lr * gradient_filter[i][j][k][n]
        self.bias = update_bias(self.bias, gradient_bias)
        return delta_prev
    
# Pooling2D    
class Pool2D:

    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, x):

        self.x = x
        self.num_filter = len(x)
        self.in_size = len(x[0])
        self.out_size = self.in_size // self.pool_size
        self.mask = zeros4(self.num_filter, self.out_size, self.out_size, self.pool_size**2)
        P = zeros3(self.num_filter, self.out_size, self.out_size)

        for j in range(self.num_filter):
            for n in range(self.out_size):
                for m in range(self.out_size):
                    window = [
                        x[j][2*n][2*m],
                        x[j][2*n+1][2*m],
                        x[j][2*n][2*m+1],
                        x[j][2*n+1][2*m+1]]

                    idx_max = max(range(4), key=lambda idx: window[idx])
                    
                    self.mask[j][n][m][idx_max] = 1
                    P[j][n][m] = window[idx_max]

        return P

    def backward(self, delta_pool):

        delta_A = zeros3(self.num_filter, self.in_size, self.in_size)

        for j in range(self.num_filter):
            for n in range(self.out_size):
                for m in range(self.out_size):
                    if self.mask[j][n][m][0]:
                        delta_A[j][2*n][2*m] = delta_pool[j][n][m]
                    if self.mask[j][n][m][1]:
                        delta_A[j][2*n+1][2*m] = delta_pool[j][n][m]
                    if self.mask[j][n][m][2]:
                        delta_A[j][2*n][2*m+1] = delta_pool[j][n][m]
                    if self.mask[j][n][m][3]:
                        delta_A[j][2*n+1][2*m+1] = delta_pool[j][n][m]

        return delta_A
    
# Linear
class Linear:

    def __init__(self, input_neurons, output_neurons):

        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.w = [[random.uniform(-0.1, 0.1) for _ in range(input_neurons)] for _ in range(output_neurons)]

    def forward(self, x):
        self.x = x
        z = [sum(x[h] * self.w[j][h] for h in range(self.input_neurons)) for j in range(self.output_neurons)]
        z = []
        for j in range(self.output_neurons):
            s = 0
            for h in range(self.input_neurons):
                s += x[h] * self.w[j][h]
            z.append(s)
        self.z = z
        return z

    def backward(self, delta):
        gradient_w = [[delta[j] * self.x[h] for h in range(self.input_neurons)] for j in range(self.output_neurons)]
        delta_prev = [sum(delta[j] * self.w[j][h] for j in range(self.output_neurons)) for h in range(self.input_neurons)]
        delta_prev = zeros1(self.input_neurons)
        for h in range(self.input_neurons):
            s = 0
            for j in range(self.output_neurons):
                s += delta[j] * self.w[j][h]
            delta_prev[h] = s

        self.w = update_weight(self.w, gradient_w)
        return delta_prev

# Softmax   
class Softmax:

    def forward(self, z):
        max_z = max(z)
        exp_values = [math.exp(v - max_z) for v in z]
        sum_exp = sum(exp_values)
        self.output = [v / sum_exp for v in exp_values]
        return self.output

    def backward(self, y_true):
        delta = [self.output[j] - y_true[j] for j in range(len(self.output))]
        return delta
    
# Flatten
class Flatten:
    
    def forward(self, x):
        flat = []
        self.x = x
        for c in x:
            for row in c:
                for v in row:
                    flat.append(v)
        return flat
    
    def backward(self, delta):
        delta_pool = []
        idx = 0
        for _ in range(len(self.x)):
            layer = []
            for _ in range(len(self.x[0])):
                row = []
                for _ in range(len(self.x[0][0])):
                    row.append(delta[idx])
                    idx += 1
                layer.append(row)
            delta_pool.append(layer)

        return delta_pool
    
# Convolutional Neural Network
class CNN:

    def __init__(self):
        self.conv1 = Conv2D(num_filter=8, filter_size=3, input_depth=1)
        self.pool1 = Pool2D(pool_size=2)
        self.conv2 = Conv2D(num_filter=16, filter_size=3, input_depth=8)
        self.pool2 = Pool2D(pool_size=2)
        self.flatten = Flatten()
        self.linear = Linear(16*5*5, 10)
        self.softmax = Softmax()

    def forward(self, x):
        x = self.conv1.forward(x) # Dim: (8, 26, 26)
        x = self.pool1.forward(x) # Dim: (8, 13, 13)
        x = self.conv2.forward(x) # Dim: (16, 11, 11)
        x = self.pool2.forward(x) # Dim: (16, 5, 5)
        self.flat = self.flatten.forward(x) # Dim: (16*5*5)
        z = self.linear.forward(self.flat) # Dim: (10)
        out = self.softmax.forward(z)
        return out

    def backward(self, y_true):


        delta = self.softmax.backward(y_true)
        delta = self.linear.backward(delta)

        delta = self.flatten.backward(delta)

        delta = self.pool2.backward(delta)
        delta = self.conv2.backward(delta)

        delta = self.pool1.backward(delta)
        delta = self.conv1.backward(delta)



cnn = CNN()

# Training and Testing

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

for epoch in range(epochs):
    train_loss_sum = 0
    test_loss_sum = 0
    correct = 0

    for i in range(len(x_train)):
        if i % 500 == 0:
            print("Training Iteration:", i)
        x = x_train[i] / 255.0
        x = [[float(v) for v in row] for row in x]
        x = [x] 
        y = y_train_oh[i]
        pred = cnn.forward(x)
        train_loss_sum += cross_entropy(pred, y)
        cnn.backward(y)


    for i in range(len(x_test)):
        x = x_test[i] / 255.0
        x = [[float(v) for v in row] for row in x]
        x = [x] 
        y = y_test_oh[i]
        pred = cnn.forward(x)
        pred_class = pred.index(max(pred))
        if pred_class == y_test[i]:
            correct += 1
        test_loss_sum += cross_entropy(pred, y)

    train_loss_sum /= len(x_train)
    test_loss_sum /= len(x_test)
    accuracy = correct / len(x_test)

    print("epoch:", epoch, "train_loss:", train_loss_sum)
    print("epoch:", epoch, "test_loss:", test_loss_sum)
    print("epoch:", epoch, "accuracy:", 100*accuracy,"%")
        

        
    


