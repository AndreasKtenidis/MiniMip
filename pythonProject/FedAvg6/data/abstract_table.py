from abc import ABC, abstractmethod


class AbstractTable(ABC):
    """
    Abstract class that defines the structure for a data.
    Subclasses must implement the methods to load data and fetch attributes.
    """
    @abstractmethod
    def  __init__(self, partition_id, num_partitions):
        pass


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
    def get_attributes(self, *attributes: str):
        pass

    @abstractmethod
    def get_attribute_names(self):
        """
        Method to return the list of attributes (keys) of the data.
        Returns:
            list: List of attribute names (e.g., column names or keys).
        """
        pass

