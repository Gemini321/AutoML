import collections
import torch
from torch.autograd import Variable
from torch.cuda import is_available
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

activation_functions = {
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'Linear': nn.Identity()
}

def create_dataset(p_val=0.1, p_test=0.2):
    import numpy as np
    import sklearn.datasets

    # Generate a dataset and plot it
    np.random.seed(0)
    num_samples = 1000

    X, y = sklearn.datasets.make_moons(num_samples, noise=0.2)
    
    train_end = int(len(X)*(1-p_val-p_test))
    val_end = int(len(X)*(1-p_test))
    
    # define train, validation, and test sets
    X_tr = X[:train_end]
    X_val = X[train_end:val_end]
    X_te = X[val_end:]

    # and labels
    y_tr = y[:train_end]
    y_val = y[train_end:val_end]
    y_te = y[val_end:]

    #plt.scatter(X_tr[:,0], X_tr[:,1], s=40, c=y_tr, cmap=plt.cm.Spectral)
    return X_tr, y_tr, X_val, y_val

class Net(nn.Module):
    # construct a new NN with given layers
    def __init__(self, num_features, num_classes, total_actions, layer_limit): 
        super(Net, self).__init__()

        max_layers = 7
        if max_layers < layer_limit:
            raise Exception('Maximum layers that ChildNet accepts is '.format(max_layers))

        # initialize class variables
        self.num_features = num_features
        self.num_classes = num_classes

        # find hidden layers
        index_eos = total_actions.index('EOS')
        hid_layers = total_actions[:index_eos]
        activations_layers = total_actions[index_eos + 1:]
        if self.num_features not in hid_layers:
            hid_layers.append(self.num_features)
        if self.num_classes not in hid_layers:
            hid_layers.append(self.num_classes)

        # initialize shared layers
        self.hid_layers = collections.defaultdict(dict)
        self.activations_layers = collections.defaultdict(dict)

        for in_dim in hid_layers:
            for out_dim in hid_layers:
                self.hid_layers[in_dim][out_dim] = nn.Linear(in_dim,
                                                             out_dim,
                                                             bias=True)
        
        for act in activations_layers:
            self.activations_layers[act] = activation_functions[act]
        
        self._hid_layers = nn.ModuleList([self.hid_layers[in_dim][out_dim]
                                                          for in_dim in hid_layers
                                                          for out_dim in hid_layers])
        self._activations_layers = nn.ModuleList([self.activations_layers[act]
                                                                          for act in activations_layers])
        self.reset_param()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
    
    def forward(self, x, layers):
        layers_added = []
        hidd_unit_prev = self.num_features

        # get shared NN layers and combine them into NN
        for i,layer in enumerate(layers):
            if isinstance(layer, int):
                layer_to_add = self.hid_layers[hidd_unit_prev][layer]
                layers_added.append(layer_to_add)
                hidd_unit_prev = layer
            elif layer == 'EOS':
                break
            else:
                layers_added.append(self.activations_layers[layer])
                
        #last layer must contain 2 out_features (2 classes)
        layers_added.append(self.hid_layers[hidd_unit_prev][self.num_classes])
        NN_layers = nn.Sequential(*layers_added)
        return NN_layers(x)

    def reset_param(self):
        init_range = 0.025
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

class ChildNet():

    def __init__(self, total_actions, layer_limit):
        self.criterion = nn.CrossEntropyLoss()

        # create dataset
        X_tr, y_tr, X_val, y_val = create_dataset()
        self.X_tr = X_tr.astype('float32')
        self.y_tr = y_tr.astype('float32')
        self.X_val = X_val.astype('float32')
        self.y_val = y_val.astype('float32')
        
        self.num_features = X_tr.shape[-1]
        self.num_classes = 2
        self.layer_limit = layer_limit

        # create shared NN
        self.net = Net(self.num_features, self.num_classes, total_actions, self.layer_limit)

    def compute_reward(self, layers, num_epochs, is_train):
        val_acc = None
        if is_train == True:
            val_acc = self.train(layers, num_epochs)
        else:
            # get validation input and expected output as torch Variables and make sure type is correct
            # Variable() without 'requires_grad=True'
            val_input = Variable(torch.from_numpy(self.X_val), requires_grad=False)
            val_targets = Variable(torch.from_numpy(self.y_val), requires_grad=False)
            val_acc = self.evaluate(layers, val_input, val_targets)
            
        return val_acc#max_val_acc#**3 #-float(val_loss.detach().numpy()) 

    def train(self, layers, num_epochs):
        # store loss and accuracy for information
        train_losses = []
        val_accuracies = []
        max_val_acc = 0
        patience = 5
        val_acc = None
        net = self.net

        # get training input and expected output as torch Variables and make sure type is correct
        tr_input = Variable(torch.from_numpy(self.X_tr), requires_grad=False)
        tr_targets = Variable(torch.from_numpy(self.y_tr), requires_grad=False)

        # get validation input and expected output as torch Variables and make sure type is correct
        # Variable() without 'requires_grad=True'
        val_input = Variable(torch.from_numpy(self.X_val), requires_grad=False)
        val_targets = Variable(torch.from_numpy(self.y_val), requires_grad=False)

        patient_count = 0
        # training loop
        for e in range(num_epochs):

            # predict by running forward pass
            tr_output = net(tr_input, layers)
            # compute cross entropy loss
            #tr_loss = F.cross_entropy(tr_output, tr_targets.type(torch.LongTensor)) 
            tr_loss = self.criterion(tr_output.float(), tr_targets.long())
            # zeroize accumulated gradients in parameters
            net.optimizer.zero_grad()
            
            # compute gradients given loss
            tr_loss.backward()
            # update the parameters given the computed gradients
            net.optimizer.step()
            
            train_losses.append(tr_loss.data.numpy())
        
            #AFTER TRAINING
            val_acc = self.evaluate(layers, val_input, val_targets)
            val_accuracies.append(val_acc)
            
            #early-stopping
            if max_val_acc > val_acc:
                patient_count += 1             
                if patient_count == patience:
                    break
            else:
                max_val_acc = val_acc
                patient_count = 0
        #print(val_accuracies)
        return val_acc

    def evaluate(self, layers, val_input, val_targets):
        net = self.net

        # predict with validation input
        val_output = net(val_input, layers)
        val_output = torch.argmax(F.softmax(val_output, dim=-1), dim=-1)
        
        # compute loss and accuracy
        #val_loss = self.criterion(val_output.float(), val_targets.long())
        val_acc = torch.mean(torch.eq(val_output, val_targets.type(torch.LongTensor)).type(torch.FloatTensor))
        
        #accuracy(val_output, val_targets)
        val_acc = float(val_acc.numpy())
        return val_acc
