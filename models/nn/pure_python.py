import math
import random
#import numpy as np
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

# Parameters

lr = 0.01
epochs = 10

# _____________Functions_____________
def zeros1(rows):
    return [0.0 for _ in range(rows)]
def zeros2(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]
def update_weight(weight, gradient,):
    for i in range(len(weight)):
        for j in range(len(weight[0])):
            weight[i][j] -= lr * gradient[i][j]
    return weight
def one_hot(y, num_classes=10):
    out = []
    for label in y:
        v = [0.0] * num_classes
        v[label] = 1.0
        out.append(v)
    return out
def cross_entropy(pred, target):
    return -sum(t * math.log(p + 1e-9) for p, t in zip(pred, target))

# ____________Classes____________
class Linear:
    def __init__(self, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.w = [[random.uniform(-0.1, 0.1) for _ in range(input_neurons)] for _ in range(output_neurons)]
        
    def forward(self, x):
        self.x = x
        self.z = [sum(x[h] * self.w[j][h] for h in range(self.input_neurons)) for j in range(self.output_neurons)]
        out = [ReLU.forward(v) for v in self.z]
        return out

    def backward(self, delta):
        for i in range(len(delta)):
            delta[i] *= ReLU.backward(self.z[i])
        gradient_w = [[delta[j] * self.x[h] for h in range(self.input_neurons)] for j in range(self.output_neurons)]
        delta_prev = [sum(delta[j] * self.w[j][h] for j in range(self.output_neurons)) for h in range(self.input_neurons)]
        self.w = update_weight(self.w, gradient_w)
        return delta_prev

class ReLU:
    @staticmethod
    def forward(x):
        return max(0, x)

    def backward(x):
        return 1 if x > 0 else 0

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



            
class NN:
    def __init__(self):
        self.layer1 = Linear(input_neurons=784, output_neurons=128)
        self.layer2 = Linear(input_neurons=128, output_neurons=64)
        self.layer3 = Linear(input_neurons=64, output_neurons=10)
        self.softmax = Softmax()
    def forward(self, x):
        self.x = x
        out = self.layer1.forward(self.x)
        out = self.layer2.forward(out)
        out = self.layer3.forward(out)
        out = self.softmax.forward(out)
        return out

    def backward(self, y_true):
        delta = self.softmax.backward(y_true)
        delta = self.layer3.backward(delta)
        delta = self.layer2.backward(delta)
        delta = self.layer1.backward(delta)




nn = NN()

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
        x = x_train[i]
        x = [float(v) for v in x]
        y = y_train_oh[i]
        pred = nn.forward(x)
        train_loss_sum += cross_entropy(pred, y)
        nn.backward(y)


    for i in range(len(x_test)):
        x = x_test[i]
        x = [float(v) for v in x]
        y = y_test_oh[i]
        pred = nn.forward(x)
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
    