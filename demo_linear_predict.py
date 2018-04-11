import numpy as np
import random as rn
import sys
import matplotlib.pyplot as plt
radius  = 1.00
center  = 0.50
pfrom   = -2.0
pto     = 2.0
epsilon = 1e-5

# our predictor is just dot product of features and weights
def predict(p, w):
    return np.dot(p, w)

# square loss for single data point p = (phi, y) for given weight vector
def loss(p, w):
    phi, y = p
    return (np.dot(phi, w) - y) ** 2

# gradient of loss for single data point where p is (phi(x), y)
def grad_loss(p, w):
    phi, y = p
    return 2.0 * (np.dot(phi, w) - y) * phi

# this is mean square error function, which we need to minimize
def training_loss(points, w):
    return sum(loss(p, w) for p in points) / len(points)

# gradient of train loss function which we use for gradient descent
# (but not for stochastic gradient descent)
def grad_train_loss(points, w):
    return sum(grad_loss(p, w) for p in points) / len(points)

# this function returns feature vector for our 'circle radius predictor'
def phi(x1, x2):
    return np.array([x1, x2, x1**2, x2**2, 1.0])

# stochastic gradient descent is less accurate, but faster than gradient_descent
def stochastic_gradient_descent(points, func_grad_of_loss = grad_loss,
                                func_train_loss = training_loss, iter = 1000, step = 0.01):
    d = len(points[0][0])
    w = np.array([0.0] * d)
    for i in range(0, iter):
        training_loss = func_train_loss(points, w)
        print("Training loss:" + str(func_train_loss(points, w)))
        sys.stdout.flush()
        if training_loss < epsilon:
            break
        for j in range(0, len(points)):
            w = w - step * func_grad_of_loss(points[j], w)
    return w

# this is slower, but more accurate gradient descent method, where we update w based on
# all points
def gradient_descent(points, gradient_of_loss = grad_train_loss,
                     func_train_loss = training_loss, iter = 1000, step = 0.01):
    d = len(points[0][0])
    w = np.array([0.0] * d)
    for i in range(0, iter):
        training_loss = func_train_loss(points, w)
        print("Training loss:" + str(func_train_loss(points, w)))
        if training_loss < epsilon:
            break
        w = w - step * gradient_of_loss(points, w)
    return w

# this function shows on the screen data. yellow points are 'positive' and other are 'negative',
# positive are those inside circle
def show_data(data, title):
    px1, px2, pc = [], [], []
    for i in range(len(data)):
        x, y = data[i]
        px1.append(x[0])
        px2.append(x[1])
        pc.append(10 if y < 0  else 200)
    plt.scatter(px1, px2, 15, c=pc, alpha=0.5)
    plt.title(title)
    plt.show()

# this function generates, and returns training data
def generate_training_data(size = 1000, show = True):
    data = []
    for n in range(size):
        x1 = rn.uniform(pfrom, pto)
        x2 = rn.uniform(pfrom, pto)
        y = radius**2 - ((x1 - center)**2 + (x2 - center)**2)
        data.append([[x1, x2], y])
    if show:
        show_data(data, "Training dataset")
    return data

# this function uses weight vector which we found during training to make predictions if
# random poinst are inside circle or not
def generate_prediction(w, size = 1000, show = True):
    data = []
    y = predict(phi(center, center), w)
    for n in range(size):
        x1 = rn.uniform(pfrom, pto)
        x2 = rn.uniform(pfrom, pto)
        y = predict(phi(x1, x2), w)
        data.append([[x1, x2], y])
    if show:
        show_data(data, "Test dataset")

# this is our main function, we generate training data, train based on them using gradient
# descent and then do predictions. we should see similar pictures for both train data and
# test data which will indicate success
def test_predictor():
    data = generate_training_data()
    points = []
    for i in range(len(data)):
        x, y = data[i]
        points.append([phi(x[0], x[1]), y])
    w = stochastic_gradient_descent(points)
    # w = gradient_descent(points)
    generate_prediction(w)

if __name__ == '__main__':
    test_predictor()