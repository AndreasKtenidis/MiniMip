
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

# Load diabetes dataset

def multivariable_regression(x,y):
    # Add intercept term
    x = sm.add_constant(x)
    # Fit OLS regression
    model = LinearRegression()
    model.fit(x,y)
    return model