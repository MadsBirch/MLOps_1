import argparse
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from data import get_data, CorruptedMNIST, get_dataloaders
from model import CNN
import utils

import matplotlib.pyplot as plt

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training model...")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=1e-3, type=float)
        parser.add_argument('--num_epochs', default=10, type=int)
        parser.add_argument('--batch_size', default=256, type=int)

        args = parser.parse_args(sys.argv[2:])
        #print(args)

        # Set device
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f'[INFO] Using device: {device}')
        
        # load data
        trainloader, _ = get_dataloaders(batch_size=args.batch_size)
        
        # load model
        model = CNN(in_channels=1, num_classes=10).to(device)
        
        # Loss and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        ### Training loop ###
        model.train()
        TRAIN_LOSS, TRAIN_ACC = [], []
         
        print(f'[INFO] Training a model for {args.num_epochs} epochs with batch size {args.batch_size} and a learning rate of {args.lr}')
        for epoch in range(args.num_epochs):
            train_loss = 0
            correct = 0
            total = 0
            for images, labels in tqdm(trainloader):
                # Get data to device if possible
                images = images.to(device=device)
                labels = labels.to(device=device)
                
                # forward
                output = model(images)
                loss = loss_fn(output, labels)
                train_loss += loss.item()

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                optimizer.step()
                
                preds = torch.argmax(F.softmax(output, dim = 1),dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            
            LOSS = train_loss/len(trainloader)
            ACC = (correct/total)*100
            
            TRAIN_LOSS.append(LOSS)
            TRAIN_ACC.append(ACC)
            
            print(f'Epoch: {epoch} | Loss: {LOSS:.5f} Acc: {ACC:.2f}%')
        
        
        utils.save_model(model=model,
                 target_dir="models",
                 model_name="first_model.pth")
        
        #return plot and accuracy
        plt.figure(figsize = (6,3))
        plt.plot(TRAIN_LOSS, label = 'train')
        plt.show()
        
        plt.figure(figsize = (6,3))
        plt.plot(TRAIN_ACC, label = 'train')
        plt.show()
        
        return LOSS, ACC
    
    def evaluate(self):
        print("Evaluating model...")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('model_path', default="")
        
        args = parser.parse_args(sys.argv[2:])
        
        # set device
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f'[INFO] Using device: {device}')
        
        # load data
        _, testloader = get_dataloaders()
        
        # load model
        model = utils.load_model(args.model_path).to(device)
        
        # loss
        loss_fn = nn.CrossEntropyLoss()     
        
        # loop over testset
        model.eval()
        correct, total = 0, 0
        train_loss, train_acc = 0,0
        
        for images, labels in testloader:
            # Get data to device if possible
            images = images.to(device=device)
            labels = labels.to(device=device)

            # forward
            output = model(images)
            loss = loss_fn(output, labels)
            train_loss += loss.item()
            
            # accuracy calc.
            preds = torch.argmax(F.softmax(output, dim = 1),dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        ACC = (correct/total)*100
        print(f'[RESULT] Model accuracy on test set: {ACC:.2f}%')
        return ACC
        

if __name__ == '__main__':
    TrainOREvaluate()


print('I am awesome')
