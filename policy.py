import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import copy
from torch.autograd import Variable

def one_hot(t, num_classes):
    '''One hot encoder of an action/hyperparameter that will be used as input for the next RNN iteration. '''
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out.astype('float32')

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

    def sample_action(self, output, batch_size, training):
        '''Stochasticity of the policy, picks a random action based on the probabilities computed by the last softmax layer. '''
        if training:
            random_array = np.random.rand(batch_size).reshape(batch_size,1)
            actions = (np.cumsum(output.detach().numpy(), axis=1) > random_array).argmax(axis=1) # sample action(return index of action)
        else: #not stochastic
            actions = (output.detach().numpy()).argmax(axis=1)
        
        for i in range(1, len(actions)):
            actions[i] = actions[i] if actions[i - 1] != 0 else 0 # if previous action is 'EOS', current action must be 'EOS'
        return actions
                
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
            action = torch.tensor(one_hot(action, self.n_outputs))            
            i += 1
            
        # prossibilities of actions of each batch, with size (batch_size, layers)
        prob = torch.stack(prob, 1)
        outputs = torch.stack(outputs, 1).squeeze(2) # confused: outputs never return?
        
        return prob, actions

    def loss(self, batch_size, action_probabilities, returns, baseline):  
        ''' Policy loss '''
        #T is the number of hyperparameters 
        sum_over_T = torch.sum(torch.log(action_probabilities.view(batch_size, -1)), axis=1)
        subs_baseline = torch.add(returns,-baseline)
        return torch.mean(torch.mul(sum_over_T, subs_baseline)) - torch.sum(torch.mul (torch.tensor(0.01) * action_probabilities, torch.log(action_probabilities.view(batch_size, -1))))

class Critic(nn.Module):
    def __init__(self, num_max_layers, num_possible_actions, n_hidden=24):
        super(Critic, self).__init__()
        self.learning_rate = 0.01
        self.num_possible_actions = num_possible_actions
        self.num_max_layers = num_max_layers
        
        self.encoder = nn.Linear(num_possible_actions, n_hidden)
        self.linear_1 = nn.Linear(num_max_layers * n_hidden, n_hidden)
        self.linear_2 = nn.Linear(n_hidden, n_hidden)
        self.linear_3 = nn.Linear(n_hidden, 1)
        self.loss_fn = nn.MSELoss(size_average=True, reduce=True)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, layers, batch_size):
        '''forward pass for a batch of data'''
        rewards = torch.Tensor(batch_size, 1)

        for i in range(batch_size):
            layer = layers[i]
            input = torch.from_numpy(one_hot(layer, self.num_possible_actions))

            h_1 = F.sigmoid(self.encoder(input)).reshape(1, -1)
            h_2 = F.leaky_relu(self.linear_1(h_1))
            h_3 = F.leaky_relu(self.linear_2(h_2))
            r = F.tanh(self.linear_3(h_3))
            rewards[i] = r
        
        return rewards

    def loss(self, y1, y2):
        '''Loss function: MSE'''
        return self.loss_fn(y1, y2)


class Policy(nn.Module):
    def __init__(self, possible_hidden_units, possible_activation_functions, layer_limit):
        super(Policy, self).__init__()
        # policy parameters
        self.possible_hidden_units = possible_hidden_units
        self.possible_activation_functions = possible_activation_functions
        self.possible_actions = possible_hidden_units + possible_activation_functions
        self.layer_limit = layer_limit
        self.train_critic_batch = 15
        self.train_critic_epochs = 10
        self.train_actor_batch = 15
        self.train_actor_epochs = 10
        self.alpha = 0.1
        self.gamma = 0.1

        # networks
        self.actor_online = PolicyNet(possible_hidden_units, possible_activation_functions, layer_limit)
        self.actor_target = copy.deepcopy(self.actor_online)
        self.critic_online = Critic( self.layer_limit, self.possible_actions)
        self.critic_target = copy.deepcopy(self.critic_online)

    def train_critic(self, layers, rewards, batch_size):
        self.critic_online.train()
        layers_target, r_target, r_online = None, None, None
        layers = Variable(torch.from_numpy(layers), requires_grad=False)
        rewards = Variable(torch.from_numpy(rewards), requires_grad=False).float()

        # compute target reward
        with torch.no_grad():
            _, layers_target = self.actor_target(batch_size, training=False)
            r_target = self.critic_target(layers_target.astype('int'), batch_size)

        # compute online reward
        r_online = self.critic_online(layers, batch_size)
        y = self.gamma * r_target + (1 - self.gamma) * rewards
        y = Variable(y, requires_grad=False)

        # compute online critic loss
        loss = self.critic_online.loss_fn(r_online, y)

        # back propagating
        self.critic_online.optimizer.zero_grad()
        loss.backward()
        self.critic_online.optimizer.step()

    def train_actor(self):
        layers, rewards = None, None

        # compute loss
        _, layers = self.actor_online(self.train_actor_batch, training=True)
        rewards = self.critic_online(torch.from_numpy(layers).int(), self.train_actor_batch)
        loss = - torch.mean(rewards)
        
        # back propagating
        self.actor_online.optimizer.zero_grad()
        loss.backward()
        self.actor_online.optimizer.step()

    def sample_actions(self, batch_size, training=False):
        return self.actor_target(batch_size, training)

    def sync_critic(self):
        self.soft_update(self.critic_target, self.critic_online, self.alpha)

    def sync_actor(self):
        self.soft_update(self.actor_target, self.actor_online, self.alpha)
    
    def soft_update(self, target, online, alpha):
        '''Soft update for target and online networks'''
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_((1 - alpha) * target_param + alpha * online_param)
    
    def train(self):
        self.actor_online.train()
        self.actor_target.train()
        self.critic_online.train()
        self.critic_online.train()

    def eval(self):
        self.actor_online.eval()
        self.actor_target.eval()
        self.critic_online.eval()
        self.critic_target.eval()