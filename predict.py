
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

def main():
    
    # get arguments from command line
    input = get_args()
    
    path_to_image = input.image_path
    checkpt = input.checkpoint
    num = input.top_k
    cat_names = input.category_names
    gpu = input.gpu
    
    # load category names file
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # load trained model
    model = load(checkpt)
    
    # Process images, predict classes, and display results
    img = Image.open(path_to_image)
    image = process_image(img)
    probs, classes = predict(path_to_image, model, num)
    check(image, path_to_image, model)
    
    
    

# Function Definitions
def get_args():
    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("image_path", type=str, help="path to image in which to predict class label")
    parser.add_argument("checkpoint", type=str, help="checkpoint in which trained model is contained")
    parser.add_argument("--top_k", type=int, default=5, help="number of classes to predict")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json",
                        help="file to convert label index to label names")
    parser.add_argument("--gpu",nargs='*', type=int, default=True,
                        help="use GPU or CPU to train model: True = GPU, False = CPU")
    
    return parser.parse_args()

# define NeuralNetwork Class with FeedForward Method
class NeuralNetwork(nn.Module):
    # define layers of the neural network: input, output, hidden layers
    def __init__(self, input_size, output_size, hidden_layers):

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

    # feedforward method    
    def forward(self, tensor):

        # Feedforward through network using relu activation function
        for linear in self.hidden_layers:
            tensor = F.relu(linear(tensor))
        tensor = self.output(tensor)

        # log_softmax: better for precision (numbers not close to 0, 1)
        return F.log_softmax(tensor, dim=1)
    
    
def load(x):
    """
        Load the saved trained model inorder to use for prediction
    """
    checkpoint = torch.load(x)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    # Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = NeuralNetwork(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'],
                             checkpoint['drop'])
    model.classifier = classifier

    
    model.classifier.load_state_dict(checkpoint['state_dict'])
    
    model.classifier.optimizer = checkpoint['optimizer']
    model.classifier.epochs = checkpoint['epochs']
    model.classifier.learning_rate = checkpoint['learning_rate']

    return model

def process_image(image):
    ''' 
        Transform an image so model can successfully predict its class.
    '''
    # resize and crop image
    w, h = image.size

    if w == h:
        size = 256, 256
    elif w > h:
        ratio = w/h
        size = 256*ratio, 256
    elif h > w:
        ratio = h/w
        size = 256, 256*ratio
        
    image.thumbnail(size, Image.ANTIALIAS)
    
    image = image.crop((size[0]//2 - 112, size[1]//2 - 112, size[0] + 112, size[1] - 112))
    
    # make color channels in between 0 and 1
    img_array = np.array(image)
    np_image = img_array/255
    
    # normalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (np_image - mean)/std
    
    # make color channel first dimension
    img = image.transpose((2, 0, 1))
    
    return img

def predict(image_path, model, num):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = Image.open(image_path)
    image = process_image(img)
    output = model(image)
    probs, indices = output.topk(topk)
    
    # convert indices to actual category names
    index_to_class = {val: key for key, val in class_to_idx.items()} #get class names from dict
    top_classes = [index_to_class[each] for each in indices]
    
    return probs, top_classes
    

def check(image, image_path, model):
    """
        Ouput a picture of the image and a graph representing its top 'k' class labels
    """
    probs, classes = predict(image_path, model)
    sb.countplot(y = classes, x = probs, color ='blue', ecolor='black', align='center')
    
    plt.show()
    ax.imsow(image)
    

    
# Run the program
if __name__ == "__main__":
    main()