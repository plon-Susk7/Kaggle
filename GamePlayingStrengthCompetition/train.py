import torch
from model import BasicNet
from dataset import CustomDataset
from torch.utils.data import DataLoader


def train_one_epoch(model,optimizer,train_loader,criterion,device):

    running_loss = 0.0
    for i,(X,y) in enumerate(train_loader):
        print(X.shape)
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred,y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss/len(train_loader)


if __name__ == '__main__':
    EPOCH = 10
    model = BasicNet()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = torch.nn.BCELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    trainLoader = DataLoader(CustomDataset('data.csv'),batch_size=32,shuffle=True)

    for epoch in range(EPOCH):

        model.train()
        avgLoss = train_one_epoch(model,optimizer,trainLoader,criterion,device)
        print(f'Epoch: {epoch+1}, Loss: {avgLoss}')


