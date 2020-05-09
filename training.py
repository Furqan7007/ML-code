import torch.nn as nn

def train(model, loader, optim, epochs):
    model.train()
    acc = 0
    criterion = nn.CrossEntropyLoss()
    count =0
    total_acc = 0
    avg_acc =0
    total_loss = 0
    avg_loss = 0

    for i, data in enumerate(loader, 0):

        data, target = data[0], data[1]
        output = model(data)
        optim.zero_grad()
        loss = criterion(output,target)
        loss.backward()
        optim.step()
        acc = (output.argmax(dim=1)==target).float().mean()
        #print("Accuracy of batch", acc*100)
        total_loss += loss
        total_acc += acc*100
        count += 1
        print("Train accuracy and loss at batch", acc.item()*100, loss.item())
    avg_acc = total_acc/count
    avg_loss = total_loss/count
    print("Acc and loss at epoch", avg_acc.numpy(), epochs)
