import torch
import torchvision
import torch.nn as nn
from torchvision.models import resnet18

sport_model = resnet18()
sport_model.fc = nn.Linear(512, 100)