import torch.nn as nn
import torch
input_size0 = 2 #28*28
hidden_size0 = 3
num_classes = 2

class Neural_net(nn.Module):
    def __init__(self, input_size, hidden_size, output):
        super(Neural_net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.relu(out)
        out = self.layer2(out)
        return out


if __name__ == '__main__':
    net = Neural_net(input_size0, hidden_size0, num_classes)
    print(net)
    print(net.layer1.weight.shape)
    print(net.layer1.weight)
    print(net.layer2.weight)






