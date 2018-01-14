import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo

class ClassificationNetwork(nn.Module):

    def __init__(self, final_categories):
        super(ClassificationNetwork, self).__init__()

        self.Res_conv = models.resnet18(pretrained = True)
        for param in self.Res_conv.parameters():
            param.requires_grad = False
        #for param in self.Res_conv.layer4.parameters():
        #    param.requires_grad = True

        self.my_model = nn.Sequential(
                        nn.Linear(1000, 500, bias=True),
                        nn.Linear(500, final_categories, bias=True),
                        )
        for param in self.my_model.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        out = self.Res_conv(x)
        out = self.my_model(out)
        return out

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print 'Saving model... %s' % path
        torch.save(self, path)
