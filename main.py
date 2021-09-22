import wandb

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import LeNet5, CustomMLP
from dataset import MNIST
from torch.utils.data import Dataset, DataLoader


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    model.train()
    model.to(device)

    trn_loss_sum = 0
    acc_sum = 0

    for i, (data, target) in enumerate(trn_loader):
        optimizer.zero_grad()
        data = data.to(device)
        outputs = model(data)
        targets = target.to(device)

        trn_loss = criterion(outputs, targets)
        trn_loss.backward()
        optimizer.step()

        wandb.log({"Train Loss": trn_loss})

        _, argmax = torch.max(outputs, dim=1)
        acc = (targets == argmax).float().mean()

        if (i + 1) % 100 == 0:
            print('Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(
                i+1, len(trn_loader), trn_loss.item(), acc.item() * 100))

        trn_loss_sum += trn_loss
        acc_sum += acc
        wandb.log({"Train Acc": acc})

    trn_loss = trn_loss_sum / len(trn_loader)
    acc = acc_sum / len(trn_loader)

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    model.eval()
    model.to(device)

    tst_loss_sum = 0
    acc_sum = 0

    with torch.no_grad():
        for data, target in tst_loader:
            data = data.to(device)
            outputs= model(data)
            targets = target.to(device)
            tst_loss = criterion(outputs, targets)
            wandb.log({"Test Loss " : tst_loss})
            tst_loss_sum += tst_loss

            output_softmax = torch.log_softmax(outputs, 1)
            _, output_tags = torch.max(output_softmax, 1)
            correct_pred = (output_tags == target).float()
            acc = correct_pred.sum() / len(correct_pred)
            acc_sum += acc * len(correct_pred)
            wandb.log({"Test Acc": acc})
    acc_sum = int(acc_sum)
    tst_loss /= len(tst_loader.dataset)
    acc = 100. * acc_sum / len(tst_loader.dataset)

    print('\nTest set: Average loss: {:.4f},Test Accuracy: {}/{} ({:.1f}%)\n'
          .format(tst_loss, acc_sum, len(tst_loader.dataset), acc))
    print('Finished Testing Test set')

    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    dir = "./data"
    wandb.init(project="mnist_lenet", entity='hohyun')

    transform = {
    'train': transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(32, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor()
    ])
    }

    batch_size = 32
    train_set = MNIST(data_dir=dir, mode='train', transform=transform['train'])
    trn_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    test_set = MNIST(data_dir=dir, mode='test', transform=transform['test'])
    tst_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 10

    criterion = nn.CrossEntropyLoss().to(device)
    # model = LeNet5().to(device)
    model = CustomMLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for i in range(epochs):
        print(f"------{i+1} epoch in progress------")
        train(model, trn_loader, device, criterion, optimizer)

    test(model, tst_loader, device, criterion)

if __name__ == '__main__':
    main()
