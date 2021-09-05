import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
from sklearn.linear_model import LinearRegression


rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y)


model = LinearRegression(fit_intercept=True)

model.fit(x[:,np.newaxis], y)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x,y)
plt.plot(xfit, yfit)

print(model.coef_)
print(model.intercept_)

rng = np.random.RandomState(1)
X = 10 * rng.rand(100,3)
Y = 0.5 + np.dot(X,[1.5, -2, 1.])

model.fit(X, Y)

print(model.intercept_)


 from sklearn.preprocessing import PolynomialFeatures

 x = np.array([2,3,4])

poly = PolynomialFeatures(3, include_bias = False)
poly.fit_transform(x[:,None])
