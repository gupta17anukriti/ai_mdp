import numpy as np
import numpy.random as npr

# binary classification : email > spam/no spam
# float classification (regressions):  location, year > price
# multiclass classification: image > cat or dog or caw
# ranking: unordered list > ordered list
#
# COMPONENTS OF LINEAR PREDICTOR
#
# 1. Feature vector function:
#
#     phi(x) = [phi_1(x), .... phi_d(x)]
#
# 2. Weight vector of the same size as feature vector, initialized with small or 0 values
#
#    w s.t. |w| == |phi(x)|
#
# 3. Score (dot product of feature vector and weight vector):
#
#    w.phi(x) = SUM(w[j] * phi(x)[j] FOR ALL j)
#
# 4. After we computed dot product of w and phi we can use:
#
#           for binary classifier Fw(x) = sign(w.phi) = [1 if w.phi >= 0 else -1]
#
# 5. Loss minimization framework
#
#       Loss(x, y, w) - we want to minimize loss
#
# 6. Margin and score:
#
#        'score' is (w.phi) - how confident we are in predicting +1
#       'margin' is (w.phi)y - how correct we are
#
# 7. Loss for binary classifier:
#
#       Loss01(x,y,w) = 1[Fw(x) != y] for classifier
#
#       for linera regression it is resedual: w.phy - y and then:
#           Loss_squared(x,y, w) = (fw(x) - y)^2
#
# 8. Training loss is average loss across antire training set
#
# Notes:
#   eta may be a constant like 0.1 or 1/sqrt(#updates made so far)
#
#   for binary predictor Loss_hidge(x,y,w) = max{1 - (w.phi)y, 0}
#           gradient: -y*phi(x) if marging less len 1 else 0
#
# grad - gradient of loss function, d - dimentionality of the problem
#
#   for multiclass classification:
#       hidge loss would be :Wy.phi(x) - max u Wu.phi(x) s.t u != y
#
def gradient_descent(grad, data, d, w = None, eta = 0.01, iters = 1000):
    if w is None:
        w = np.zeros(d)
    for i in range(iters):
        w = w - eta * grad(data, w)
    return w

def stochastic_gradient_descent(grad, data, d, w = None, eta = 0.01, iters = 100):
    if w is None:
        w = np.zeros(d)
    for i in range(iters):
        for p in data:
            w = w - eta * grad(p, w)
    return w

def G(points, w):
    return sum(2*(np.dot(w, x) - y) * x for x, y in points) / len(points)

def sG(point, w):
    x, y = point
    return 2*(w.dot(x) - y) * x

def run_test(dfunc, gf):
    true_w = [1, 2, 3, 4, 5]
    d, points = len(true_w), []
    for t in range(1000):
        x = npr.randn(d)
        y = x.dot(true_w) + npr.randn()
        points.append((x, y))
    w = dfunc(gf, points, d)
    print(w)

def test_gradient_descent():
    run_test(gradient_descent, G)

def test_stochastic_gradient_descent():
    run_test(stochastic_gradient_descent, sG)

if __name__ == "__main__":
    test_gradient_descent()
    test_stochastic_gradient_descent()



