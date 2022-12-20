# -*-coding:Utf-8 -*
import json


class Config:
    """
    Object containing configuration information
    """

    def __init__(self, path):
        """
        Initialization method
        """
        self.path = path
        self.domain = None
        self.features = None

    def read(self):
        """
        Read configuration file stored in self.path.
        """
        # Read
        conf = json.load(open(self.path, "rb"))
        self.domain = conf["computation_domains"]
        self.features = conf["features"]
