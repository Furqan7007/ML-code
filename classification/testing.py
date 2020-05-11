import torch.nn as nn

def test(model, loader, optim, epoch):
    model.eval()
    acc = 0
    criterion = nn.CrossEntropyLoss()
    count = 0
    total_acc = 0
    avg_acc = 0 

    for i, data in enumerate(loader,0):
        data,target = data[0], data[1]
        output = model(data)
        loss = criterion(output, target)
        acc = (output.argmax(dim=1)==target).float().mean()
        #print("Test Accuracy", acc*100)
        total_acc += acc
        count +=1
    avg_acc = total_acc/count
    print("Test accuracy at epoch", avg_acc, epoch)
