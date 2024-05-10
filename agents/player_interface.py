from abc import ABC, abstractmethod
from faker import Faker


class Player(ABC):
    """Abstract Base Class for a Player"""

    @abstractmethod
    def __init__(self, name=Faker().name()):
        """Initialization of a player"""
        self.name = name

    @abstractmethod
    def action(self, action_space, observation, info):
        """Method that calculates the move based on the observation array and the action space."""
        pass
