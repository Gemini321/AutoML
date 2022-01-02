from numpy.lib.type_check import nan_to_num
import torch
from torch.nn.modules import loss
from childNet import ChildNet
from utils import fill_tensor, indexes_to_actions
from torch.autograd import Variable
import numpy as np

class Trainer(object):
    def __init__(self, policy, batch_size, total_actions, train_shared_epochs, verbose=False, num_episodes=50):
        # training settings
        self.batch_size = batch_size
        self.total_actions = total_actions
        self.train_shared_epochs = train_shared_epochs
        self.verbose = verbose
        self.num_episodes = num_episodes
        self.train_shared_epochs = 5
        self.train_shared_NN_epochs = 100
        self.train_shared_batch_size = 5
        self.train_controller_epochs = 10
        self.decay = 0.9
        self.val_freq = 1

        self.policy = policy
        self.shared = ChildNet(total_actions, policy.layer_limit)

    def training(self):
        ''' Optimization/training loop of the policy net. Returns the trained policy. '''
        
        # train policy network
        training_rewards, val_rewards, losses = [], [], []
        baseline = torch.zeros(15, dtype=torch.float)
        
        # start training for num_episodes episodes
        print('start training')
        for i in range(self.num_episodes):
            print('Epoch {}:'.format(i))
            
            self.train_shared()

            baseline, training_rewards, losses = self.train_controller(baseline, training_rewards, losses)
            
            # print training
            if self.verbose:
                print('{:4d}. mean training reward: {:6.2f}, mean loss: {:7.4f}'.
                    format(i+1, np.mean(training_rewards[-self.val_freq:]), np.mean(losses[-self.val_freq:])))
                print('{:4d}. max training reward: {:6.2f}, min loss: {:7.4f}'.
                    format(i+1, np.max(training_rewards[-self.val_freq:]), np.min(losses[-self.val_freq:])))

                #batch_hid_units, _ = indexes_to_actions(self.policy(batch_size=self.batch_size, training=True)[0], self.batch_size, self.total_actions)
                #print('{:4d}. best network: {}'.format(i+1, batch_hid_units[0]))
            print()

        print('done training')  
    
        return self.policy

    def train_controller(self, baseline, training_rewards, losses):
        cn = self.shared
        self.policy.train()
        cn.net.eval()

        for epochs in range(self.train_controller_epochs):
            print('{:4d}th controller training:'.format(epochs))
            batch_r, batch_a_probs = [], []
            prob, actions = self.policy(self.batch_size, training=True)
            batch_hid_units, batch_index_eos = indexes_to_actions(actions, self.batch_size, self.total_actions)

            #compute individually the rewards
            for j in range(self.batch_size):
                # policy gradient update 
                if self.verbose:
                    print(batch_hid_units[j])
                # train child network and compute reward
                if batch_hid_units[j] == ['EOS']:
                    r = 0.5
                else:
                    r = cn.compute_reward(batch_hid_units[j], self.train_shared_epochs, is_train=False)
                a_probs = prob[j, :batch_index_eos[j] + 1]

                batch_r += [r]
                batch_a_probs += [a_probs.view(1, -1)]

            #rearrange the action probabilities
            a_probs = []
            for b in range(self.batch_size):
                a_probs.append(fill_tensor(batch_a_probs[b], self.policy.n_outputs, ones=True))
            a_probs = torch.stack(a_probs,0)

            #convert to pytorch tensors --> use get_variable from utils if training in GPU
            batch_a_probs = Variable(a_probs, requires_grad=True)
            batch_r = Variable(torch.tensor(batch_r), requires_grad=True)
            
            # classic traininng steps
            loss = self.policy.loss(self.batch_size, batch_a_probs, batch_r, torch.mean(baseline))
            self.policy.optimizer.zero_grad()  
            loss.backward()
            self.policy.optimizer.step()

            # actualize baseline
            baseline = torch.cat((baseline[1:]*self.decay, torch.tensor([torch.mean(batch_r)*(1-self.decay)], dtype=torch.float)))
            
            # bookkeeping
            training_rewards.append(torch.mean(batch_r).detach().numpy())
            losses.append(loss.item())

        return baseline, training_rewards, losses

    def train_shared(self):
        cn = self.shared
        cn.net.train()
        self.policy.eval()

        for epochs in range(self.train_shared_epochs):
            print('{:4d}th shared training:'.format(epochs))
            batch_r = []
            batch_a_probs =[]

            with torch.no_grad():
                prob, actions = self.policy(self.train_shared_batch_size, training=True)
            batch_hid_units, batch_index_eos = indexes_to_actions(actions, self.train_shared_batch_size, self.total_actions)

            #compute individually the rewards
            for j in range(self.train_shared_batch_size):
                # policy gradient update 
                if self.verbose:
                    print(batch_hid_units[j])
                # train child network and compute reward
                r = cn.compute_reward(batch_hid_units[j], self.train_shared_NN_epochs, is_train=True)
                a_probs = prob[j, :batch_index_eos[j] + 1]

                batch_r += [r]
                batch_a_probs += [a_probs.view(1, -1)]
