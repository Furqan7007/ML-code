import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
from training import train
from testing import test
import argparse
from args import parser

def main():

    args = parser.parse_args()

    dataset = args.dataset
    run_name = args.dataset+args.model if args.name=="" else args.name

    if(dataset == "MNIST"):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size = args.batch_size, shuffle = True)

    if(dataset == "CIFAR10"):
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size = args.batch_size, shuffle = True)

    use_cuda = args.use_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    if(args.model == "MobileNetV2"):
        if(dataset == "CIFAR10"):
            net = torchvision.models.MobileNetV2(num_classes=10)
        if(dataset == "CIFAR100"):
            net = torchvision.models.MobileNetV2(num_classes=100)
        
    
    optimizer = optim.SGD(net.parameters(),lr=0.01)
    epochs = 50

    for i in tqdm(range(epochs)):
        train(net, train_loader, optimizer, i)
        test(net, test_loader, optimizer, i)

    torch.save(net.state_dict(), run_name+".pt")

if __name__ == '__main__':
    main()
