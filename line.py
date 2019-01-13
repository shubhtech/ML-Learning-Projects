from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt

from matplotlib import style
style.use('fivethirtyeight')

x = np.array([1,2,3,4,5])
y = np.array([5,4,6,5,6])

#plt.plot(x,y)
#plt.show()

def best_fit_slope_intercept(x, y):
    m = (((mean(x) * mean(y)) - mean(x*y) ) /
    ((mean(x) * mean(x)) - mean(x*x))  )
    b = mean(y) - m*mean(x)
    return m,b

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

m,b = best_fit_slope_intercept(x,y)
#print (m,b)

regression_line = [(m*i)+b for i in x]

#for i in x:
#    regression_line.append((m*x)+b)

predict_x = 8
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(y, regression_line)
print (r_squared)


plt.scatter(x,y)
plt.scatter(predict_x, predict_x, color='r')
plt.plot(x, regression_line)
plt.show()
