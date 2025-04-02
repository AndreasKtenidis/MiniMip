from data.pandas.pandas_column import PandasColumn
from data.pandas.pandas_dataset import PandasDataset

dataset = PandasDataset(2,3)
col=PandasColumn(dataset,'SepalWidthCm')
col2=PandasColumn(dataset,'SepalLengthCm')
col3= col+col2
print(col.vec)
print(col2.vec)
print(col3.vec)