import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """Policy network, i.e., RNN controller that generates the different childNet architectures."""

    def __init__(self, possible_hidden_units, possible_activation_functions, layer_limit):
        super(PolicyNet, self).__init__()
        
        # parameters
        self.layer_limit = layer_limit
        self.gamma = 1.0
        self.n_hidden = 24
        self.possible_hidden_units = possible_hidden_units
        self.possible_activation_functions = possible_activation_functions
        self.n_outputs = possible_hidden_units + possible_activation_functions
        self.learning_rate = 1e-2
        
        # Neural Network
        self.lstm_1 = nn.LSTMCell(self.n_outputs, self.n_hidden)
        self.lstm_2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, self.n_outputs)

        # training
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def one_hot(self, t, num_classes):
        '''One hot encoder of an action/hyperparameter that will be used as input for the next RNN iteration. '''
        out = np.zeros((t.shape[0], num_classes))
        for row, col in enumerate(t):
            out[row, col] = 1
        return out.astype('float32')

    def sample_action(self, output, batch_size, training):
        '''Stochasticity of the policy, picks a random action based on the probabilities computed by the last softmax layer. '''
        if training:
            random_array = np.random.rand(batch_size).reshape(batch_size,1)
            return (np.cumsum(output.detach().numpy(), axis=1) > random_array).argmax(axis=1) # sample action(return index of action)
        else: #not stochastic
            return (output.detach().numpy()).argmax(axis=1)
                
    def forward(self, batch_size, training):
        ''' Forward pass. Generates different childNet architectures (nb of architectures = batch_size). '''
        outputs = []
        prob = []
        actions = np.zeros((batch_size, self.layer_limit))
        # confused: action is set by torch.zeros() before any operation
        action = not None #initialize action to don't break the while condition
        i = 0
        counter_nb_layers = 0
        
        # LSTM input(default zeros)
        h_t_1 = torch.zeros(batch_size, self.n_hidden, dtype=torch.float)
        c_t_1 = torch.zeros(batch_size, self.n_hidden, dtype=torch.float)
        h_t_2 = torch.zeros(batch_size, self.n_hidden, dtype=torch.float)
        c_t_2 = torch.zeros(batch_size, self.n_hidden, dtype=torch.float)
        action = torch.zeros(batch_size, self.n_outputs, dtype=torch.float)
        
        # for each layer of DNN(action chosen in units numbers and activation functions randomly?)
        while counter_nb_layers<self.layer_limit: 

            h_t_1, c_t_1 = self.lstm_1(action, (h_t_1, c_t_1))
            #h_t_2, c_t_2 = self.lstm_2(h_t_1, (h_t_2, c_t_2))
                        
            # when layer i is even, set the possibilities of activation functions to zero
            # when layer i is odd, set the possibilities of full connected layers to zero
            output = F.softmax(self.linear(h_t_1))
            if i % 2 == 0:
                output[:, self.possible_hidden_units:] = 0
            else:
                output[:, :self.possible_hidden_units] = 0
            output = output / output.sum(dim=1).unsqueeze(dim=1)
            counter_nb_layers += 1
            action = self.sample_action(output, batch_size, training)

            # collect prosibilities of each action and chosen action
            outputs += [output]
            prob.append(output[np.arange(batch_size),action])
            actions[:, i] = action
            action = torch.tensor(self.one_hot(action, self.n_outputs))            
            i += 1
            
        # prossibilities of actions of each batch, with size (batch_size, layers)
        prob = torch.stack(prob, 1)
        outputs = torch.stack(outputs, 1).squeeze(2) # confused: outputs never return?
        
        return prob, actions

    def loss(self, batch_size, action_probabilities, returns, baseline):  
        ''' Policy loss. More details see the article uploaded in https://github.com/RualPerez/AutoML '''
        #T is the number of hyperparameters 
        sum_over_T = torch.sum(torch.log(action_probabilities.view(batch_size, -1)), axis=1)
        subs_baseline = torch.add(returns,-baseline)
        return torch.mean(torch.mul(sum_over_T, subs_baseline)) - torch.sum(torch.mul (torch.tensor(0.01) * action_probabilities, torch.log(action_probabilities.view(batch_size, -1))))
