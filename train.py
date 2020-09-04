"""
    Will train a particular dataset using a pretrained model. User can input which pretrained model to use,
    number of epochs, learning rate, and number of hidden layers.
    Then accuracy, training loss, and validation loss will be printed.
"""
# Imports
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image


# load pretrained models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

def main():
    
    # get arguments from command line
    input = get_args()
    
    data_dir = input.data_directory
    save_to = input.save_dir
    pretrained_model = input.arch
    learning_rate = input.learning_rate
    ep = input.epochs
    hidden_layers = input.hidden_units
    output_size = input.output
    gpu = input.gpu
    drop = 0.2
    
    # load and process images from data directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    trainloader, validloader, testloader = process(train_dir, valid_dir, test_dir)
    
    # load the pretrained model and determine input size
    model_dict = {"vgg": vgg16, "resnet": resnet18, "alexnet": alexnet}
    inputsize_dict = {"vgg": 25088, "resnet": 512, "alexnet": 9216}
    
    model = model_dict[pretrained_model]
    input_size = inputsize_dict[pretrained_model]
    
    # Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # define criterion and optimizer
    classifier = NeuralNetwork(input_size, output_size, hidden_layers, drop)
    if pretrained_model in ['vgg', 'alexnet']:
        model.classifier = classifier
    else:
        model.fc = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Train model
    print("Training Loss:")
    train(model, trainloader, validloader, criterion, optimizer, ep, gpu)
    
    # Print Accuracy
    test_loss = accuracy(testloader, model, criterion, gpu)
    print("Accuracy (using test data):")
    print(test_loss)
    
    # Save Checkpoint
    
    # Save Checkpoint: input, output, hidden layer, epochs, learning rate, model
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'drop': drop,
                  'epochs': ep,
                  'learning_rate': learning_rate,
                  'arch': pretrained_model,
                  'optimizer': optimizer,
                  'state_dict': model.classifier.state_dict()}

    torch.save(checkpoint, save_to)

# Function Definitions
def get_args():
    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_directory", type=str, help="data directory containing training and testing data")
    parser.add_argument("--save_dir", type=str, default="checkpoint_2.pth",
                        help="directory where to save trained model and hyperparameters")
    parser.add_argument("--arch", type=str, default="vgg16",
                        help="pre-trained model: vgg16, resnet18, alexnet")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                        help="number of epochs to train model")
    parser.add_argument("--hidden_units",nargs='*', type=int, default=[700, 300],
                        help="list of hidden layers")
    parser.add_argument('--gpu', action='store_true', help='Use this argument if you want to train on GPU.')
    parser.add_argument("--output", type=int, default=102,
                        help="enter output size")
    
    return parser.parse_args()

def process(train_dir, valid_dir, test_dir):
    """
        Load and transform data so pretrained model can successfully train.
    """
    train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    return trainloader, validloader, testloader

class NeuralNetwork(nn.Module):
    
    # define layers of the neural network: input, output, hidden layers, dropout
    def __init__(self, input_size, output_size, hidden_layers, drop):
        
        # calls init method of nn.Module (base class)
        super().__init__()
        
        # input_size to hidden_layer 1 : ModuleList --> list meant to store nn.Modules
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # add arbitrary number of hidden layers
        i = 0
        j = len(hidden_layers)-1
        
        while i != j:
            l = [hidden_layers[i], hidden_layers[i+1]]
            self.hidden_layers.append(nn.Linear(l[0], l[1]))
            i+=1

        # check to make sure hidden layers formatted correctly
        for each in hidden_layers:
            print(each)
        
        # last hidden layter -> output
        self.output = nn.Linear(hidden_layers[j], output_size)
        self.dropout = nn.Dropout(p = drop)
        
    # feedforward method    
    def forward(self, tensor):
        
        # Feedforward through network using relu activation function
        for linear in self.hidden_layers:
            tensor = F.relu(linear(tensor))
            tensor = self.dropout(tensor)
        tensor = self.output(tensor)
        
        # log_softmax: better for precision (numbers not close to 0, 1)
        return F.log_softmax(tensor, dim=1)



        
# Train function: will use GPU (faster)
def train(model, trainloader, validloader, criterion, optimizer, epochs, gpu):
    
    valid_len = len(validloader)
    print_every = 100
    #model.train()
    
    steps = 0
    
    # change to cuda
    if gpu == True:
        model.to('cuda')
    e = 0

    while e < epochs:
        running_loss = 0
        val_loss = 0
        for inputs, labels in iter(trainloader):
            steps += 1
            if gpu==True:
                device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
                inputs, labels = inputs.to(device), labels.to(device)
                

            # don't want to sum up all gradients for each epoch
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            


            if steps % print_every == 0:
                # inference
                model.eval()
                val_loss, accuracy = valid(criterion, model, validloader, device)
                
                print("Epoch:{}/{}".format(e+1, epochs), "... Loss:{}".format(running_loss/print_every),
                      "Validation Loss:{}".format(val_loss/valid_len),
                      "Validation Accuracy:{}".format(accuracy))
                running_loss = 0
                model.train()
            
        # next epoch
        e = e + 1
                    
def accuracy(testloader, model, criterion, gpu):
    """
        Compute how accurate the trained model predicts data it was not trained on
    """
    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if gpu==True:
                device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
                images, labels = inputs.to(device), labels.to(device)
                images, labels = images.to('cuda'), labels.to('cuda')

            #forward
            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            #tensor with probability, tensor with index of flower category
            prob = torch.exp(output) #tensor with prob. of each flower category
            pred = prob.max(dim=1) #tensor giving us flower label most likely

            #calculate number correct
            matches = (pred[1] == labels.data)
            correct += matches.sum().item()
            total += 64

        acc = 100*(correct/total)
        return acc
    
# Validation: accuracy and validation set loss
def valid(criterion, model, validloader, device):
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for data in validloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            #forward
            output = model.forward(images)
            val_loss += criterion(output, labels).item()

            #tensor with probability, tensor with index of flower category
            prob = torch.exp(output) #tensor with prob. of each flower category
            pred = prob.max(dim=1) #tensor giving us flower label most likely

            #calculate number correct
            matches = (pred[1] == labels.data)
            correct += matches.sum().item()
            total += 64
            accuracy = 100*(correct/total)

    return val_loss, accuracy
if __name__ == "__main__":
    main()
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
