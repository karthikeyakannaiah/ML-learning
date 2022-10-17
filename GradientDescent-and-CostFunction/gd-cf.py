# gradient descent and cost function.
# mean squared error
import numpy as np
import math
def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.01
    prev_cost = 0
    for i in range(iterations):
        y_pred = m_curr*x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_pred)])
        if i!=0 and (math.isclose(prev_cost,cost) or math.isclose(cost, math.inf)):
            print('iteration',i)
            break
        prev_cost = cost
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum(y-y_pred)
        m_curr = m_curr - md * learning_rate
        b_curr = b_curr - bd * learning_rate
        print('m {}, b {}, cost {}, iteration {}'.format(m_curr,b_curr,cost,i))

x = np.array([92,56,88,70,80,49,65,35,66,67])
y = np.array([98,68,81,80,83,52,66,30,68,73])

gradient_descent(x,y)
