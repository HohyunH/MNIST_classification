import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, img):
        x = self.feature_extractor(img)
        x = torch.flatten(x, 1)

        logit = self.classifier(x)
        output = F.softmax(logit, dim=1)
        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super(CustomMLP, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(32 * 32, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 10),
            nn.Softmax(dim=-1))

    def forward(self, img):
        x = torch.flatten(img, 1)
        x = self.feature_extractor(x)
        return x

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = CustomMLP().to(device)

    summary(model, (1, 32, 32))
