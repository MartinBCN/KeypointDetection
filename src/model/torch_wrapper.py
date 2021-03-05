from pathlib import Path
from typing import Union, Callable
import torch
from torch import nn
import torch.optim as optim
from torch.nn import Module


class TorchWrapper:
    def __init__(self):

        self.model = None

        self.criterion = None
        self.optimizer = None

        self.training_log = {}

    def set_criterion(self, criterion: Module):
        self.criterion = criterion

    def set_optimizer(self, optimizer: Callable, hyper_parameter: dict = {}):
        self.optimizer = optimizer(self.model.parameters(), **hyper_parameter)

    @classmethod
    def load(cls, filepath: Union[str, Path]):
        state = torch.load(filepath)

        new = cls()

        new.set_optimizer()

        new.model.load_state_dict(state['model_state_dict'])
        new.optimizer.load_state_dict(state['optimizer_state_dict'])
        new.training_log = state['training_log']

        return new

    def save(self, filepath: Union[str, Path]):

        state = {'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'training_log': self.training_log
                 }

        torch.save(state, filepath)

