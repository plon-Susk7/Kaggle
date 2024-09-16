import torch.nn as nn 

class BasicNet(nn.Module):

    def __init__(self):
        super(BasicNet,self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(2,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        return self.layer(x)