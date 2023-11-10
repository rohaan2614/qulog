"""
    A Model of the Neural Network

    Returns:
        _type_: Torch Class Object
"""
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    """
        Model of the neural network. Inherits Torch neural network and used in conjunction with utils
    """
    def __init__(self, in_features, h1, h2, out_features=3):
        super().__init__()
        self.in_features = in_features
        self.h1 = h1
        self.h2 = h2
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        """
        The forward function applies softmax activation to the input x and passes it through two fully
        connected layers and an output layer.

        :param x: The parameter `x` represents the input to the forward method. It is passed through two
        fully connected layers (`self.fc1` and `self.fc2`) with softmax activation functions applied
        after each layer. Finally, the output is passed through the `self.out` layer and returned
        :return: The output of the last layer, `x`, is being returned.
        """
        x = F.softmax(self.fc1(x))
        x = F.softmax(self.fc2(x))
        x = self.out(x)
        return x