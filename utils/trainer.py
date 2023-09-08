import torch
from utils import plots

def train(net, optimizer, criterion, scheduler, train_loader, test_loader, epochs):
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        _train_cycle(net, optimizer, criterion, train_loader, train_losses, epoch, epochs)
        _test_cycle(net, optimizer, criterion, test_loader, test_losses, epoch)
        scheduler.step()

    plots.draw_loss_graphs(train_losses, test_losses)

def _train_cycle(net, optimizer, criterion, train_loader, train_losses, epoch, epochs):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

    train_losses.append(train_loss)

    print("Epoch: {}/{} ––".format(epoch + 1, epochs),
            "Train loss: {:.3f} ––".format(train_loss),
            "Train accuracy: {:.3f} –– ".format(correct / total),
            end = "")

def _test_cycle(net, optimizer, criterion, test_loader, test_losses, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

    test_losses.append(test_loss)
            
    print("Test loss: {:.3f} ––".format(test_loss),
            "Test accuracy: {:.3f}".format(correct / total))