import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


# Hyperparameters
num_epochs = 2
batch_size = 10
learning_rate = 0.001

# Dataset
# Transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Training data
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False)

# Classes
classes = ('airplane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Initialize 3x3 kernel with random weights between -1 and 1, inclusive
w = np.random.uniform(low=-1, high=1, size=(3, 3))


# Loss- Cross Entropy Loss
def criterion(y, y_pred):
    pass


# Optimizer - Stochastic Gradient Descent
def optimizer(model, lr):
    pass


# Gradients
def gradient(x, y, y_pred):
    pass


# Convolutional Neural Network
class ConvNet():
    def __init__(self):
        super(ConvNet, self).__init__()
        pass

    def forward(self, x):
        pass


model = ConvNet()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        # Forward pass
        y_pred = ConvNet().forward(images)
        loss = criterion(y_pred, labels)

        # Backward Pass

        # Update weights
