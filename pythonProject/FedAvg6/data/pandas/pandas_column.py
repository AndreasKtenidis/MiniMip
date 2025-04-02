from data.column import Column
from data.pandas.pandas_dataset import PandasDataset
from numbers import Number


class PandasColumn(Column):

    def __init__(self, dataset2: PandasDataset, attribute):
        self.vec = dataset2.get_attribute(attribute)
        self.type = dataset2.get_type(attribute)
        self.attribute = attribute

    def __init__(self, dataset2: PandasDataset, attribute):
        self.vec = dataset2.get_attribute(attribute)
        self.type = dataset2.get_type(attribute)
        self.attribute = attribute

    def __init__(self, dataset2: PandasDataset, attribute):
        self.vec = dataset2.get_attribute(attribute)
        self.type = dataset2.get_type(attribute)
        self.attribute = attribute

    def get_type(self):
        return self.type

    def get_name(self):
        return self.attribute

    def len(self):
        return len(self.vec)

    def __add__(self, other):
        """
        Add a Column to another Column or a number to a Column.

        Args:
            other (Column or Number): Another Column instance or a number (int, float).

        Returns:
            Column: A new Column instance with the added data.
        """
        if isinstance(other, PandasColumn):
            self.vec=self.add_column(other)
        elif isinstance(other, Number):
            self.vec=self.vec +other
        else:
            raise ValueError(f"Cannot add Column with object of type {type(other)}")

    def add_column(self,column):
        if len(self.vec) != len(column.vec):
            raise ValueError("Columns must have the same length to add them.")
        return self.vec+column.vec


