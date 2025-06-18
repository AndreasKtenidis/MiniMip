import numpy as np


class ArrayWrapper:
    def __init__(self, data, client=None):
        """
        Initialize the wrapper with a NumPy array or array-like data.

        Parameters:
        -----------
        data : array_like
            Input data that can be converted to a NumPy array
        client : object, optional
            A client object associated with this array (e.g., for remote operations)
        """
        self._array = np.asarray(data)
        self._client = client

    @property
    def array(self):
        """Get the underlying NumPy array"""
        return self._array

    @array.setter
    def array(self, value):
        """Set the underlying NumPy array"""
        self._array = np.asarray(value)

    @property
    def client(self):
        """Get the client object"""
        return self._client

    @client.setter
    def client(self, value):
        """Set the client object"""
        self._client = value

    @property
    def shape(self):
        """Get the shape of the array"""
        return self._array.shape

    @property
    def dtype(self):
        """Get the data type of the array"""
        return self._array.dtype

    def __repr__(self):
        """String representation of the wrapper"""
        client_repr = f", client={self._client!r}" if self._client is not None else ""
        return f"ArrayWrapper({self._array.__repr__()}{client_repr})"

    def __str__(self):
        """String representation for printing"""
        client_str = f"\nClient: {self._client}" if self._client is not None else ""
        return f"Wrapped Array:\n{self._array.__str__()}{client_str}"

    def __getitem__(self, key):
        """Indexing support"""
        return ArrayWrapper(self._array[key], client=self._client)

    def __setitem__(self, key, value):
        """Assignment support"""
        self._array[key] = value

    def __len__(self):
        """Length of the array"""
        return len(self._array)

    def __array__(self):
        """Support for NumPy's array interface"""
        return self._array

    # Math operations (all return new ArrayWrapper with same client)
    def __add__(self, other):
        if isinstance(other, ArrayWrapper):
            return ArrayWrapper(self._array + other._array, client=self._client)
        return ArrayWrapper(self._array + other, client=self._client)

    def __sub__(self, other):
        if isinstance(other, ArrayWrapper):
            return ArrayWrapper(self._array - other._array, client=self._client)
        return ArrayWrapper(self._array - other, client=self._client)

    def __mul__(self, other):
        if isinstance(other, ArrayWrapper):
            return ArrayWrapper(self._array * other._array, client=self._client)
        return ArrayWrapper(self._array * other, client=self._client)

    def __truediv__(self, other):
        if isinstance(other, ArrayWrapper):
            return ArrayWrapper(self._array / other._array, client=self._client)
        return ArrayWrapper(self._array / other, client=self._client)

    # Inequality operations
    def __lt__(self, other):
        """Less than operation <"""
        if isinstance(other, ArrayWrapper):
            return ArrayWrapper(self._array < other._array, client=self._client)
        return ArrayWrapper(self._array < other, client=self._client)

    def __gt__(self, other):
        """Greater than operation >"""
        if isinstance(other, ArrayWrapper):
            return ArrayWrapper(self._array > other._array, client=self._client)
        return ArrayWrapper(self._array > other, client=self._client)

    def __le__(self, other):
        """Less than or equal operation <="""
        if isinstance(other, ArrayWrapper):
            return ArrayWrapper(self._array <= other._array, client=self._client)
        return ArrayWrapper(self._array <= other, client=self._client)

    def __ge__(self, other):
        """Greater than or equal operation >="""
        if isinstance(other, ArrayWrapper):
            return ArrayWrapper(self._array >= other._array, client=self._client)
        return ArrayWrapper(self._array >= other, client=self._client)

    def __eq__(self, other):
        """Equality operation =="""
        if isinstance(other, ArrayWrapper):
            return ArrayWrapper(self._array == other._array, client=self._client)
        return ArrayWrapper(self._array == other, client=self._client)

    def __ne__(self, other):
        """Not equal operation !="""
        if isinstance(other, ArrayWrapper):
            return ArrayWrapper(self._array != other._array, client=self._client)
        return ArrayWrapper(self._array != other, client=self._client)

    # Additional utility methods
    def describe(self):
        """Print descriptive statistics about the array"""
        print(f"Array shape: {self.shape}")
        print(f"Data type: {self.dtype}")
        if self._client is not None:
            print(f"Client: {self._client}")
        if self._array.ndim > 0 and self._array.size > 0:
            print(f"Min: {np.min(self._array)}")
            print(f"Max: {np.max(self._array)}")
            print(f"Mean: {np.mean(self._array)}")
            print(f"Std: {np.std(self._array)}")

    def apply(self, func):
        """Apply a function to the array and return a new wrapped array"""
        return ArrayWrapper(func(self._array), client=self._client)

    def to_list(self):
        """Convert the array to a Python list"""
        return self._array.tolist()


    def min(self):
        # TODO !!!
        return self._array.min()

    def dot(self,w):
        return self._array.dot(w)

    def where(self, condition, other=None):
        """
        Return elements chosen from self or other depending on condition.

        Parameters:
        -----------
        condition : ArrayWrapper or array_like
            Where True, yield self, otherwise yield other
        other : scalar or ArrayWrapper or array_like, optional
            Value(s) to use when condition is False
        """
        if isinstance(condition, ArrayWrapper):
            condition = condition.array
        if isinstance(other, ArrayWrapper):
            other = other.array
        return ArrayWrapper(np.where(condition, self._array, other), client=self._client)