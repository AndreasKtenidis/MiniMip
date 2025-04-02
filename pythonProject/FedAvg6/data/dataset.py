from abc import ABC, abstractmethod
from numbers import Number  # To handle numeric types like int, float


class Dataset(ABC):
    """
    Abstract class that defines the structure for a data.
    Subclasses must implement the methods to load data and fetch attributes.
    """


    @abstractmethod
    def get_attribute(self, attribute):
        """
        Method to retrieve a specific attribute from the data.
        Args:
            attribute (str): The attribute name to fetch.
        Returns:
            list or ndarray: The values of the attribute.
        """
        pass

    @abstractmethod
    def get_type(self, attribute):
        """
        Method to return the type of the specified attribute.
        Args:
            attribute (str): The attribute name to fetch the type for.
        Returns:
            type: The type of the attribute (e.g., int, float).
        """
        pass

    @abstractmethod
    def get_attributes(self):
        """
        Method to return the list of attributes (keys) of the data.
        Returns:
            list: List of attribute names (e.g., column names or keys).
        """
        pass

