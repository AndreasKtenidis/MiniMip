
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd

# Load iris dataset and convert to DataFrame
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)



def smd(df):
    # Choose two attributes for comparison
    attr1 = df['sepal length (cm)']
    attr2 = df['petal width (cm)']

    # Compute means
    mean1 = attr1.mean()
    mean2 = attr2.mean()

    # Compute sample standard deviations
    std1 = attr1.std(ddof=1)
    std2 = attr2.std(ddof=1)

    # Compute pooled standard deviation
    pooled_std = np.sqrt((std1 ** 2 + std2 ** 2) / 2)

    # Compute SMD
    smd = (mean1 - mean2) / pooled_std

    print(f"Standardized Mean Difference (SMD): {smd:.4f}")


smd(df)


