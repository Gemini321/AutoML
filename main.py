from policy import Policy
from training import Trainer 
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch 

if __name__ == "__main__":
        
    # input parameters
    parser = argparse.ArgumentParser(description='Documentation in the following link: https://github.com/RualPerez/AutoML', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--batch', help='Batch size of the policy (int)', nargs='?', const=1, type=int, default=1)
    parser.add_argument('--max_layer', help='Maximum nb layers of the childNet (int)', nargs='?', const=1, type=int, default=6)
    parser.add_argument('--possible_hidden_units', default=[1,8,32], nargs='*',  # not [1,2,4,8,16,32]
                        type=int, help='Possible hidden units of the childnet (list of int)')
    parser.add_argument('--possible_act_functions', default=['Tanh', 'ReLU'], nargs='*', # not ['Sigmoid', 'Tanh', 'ReLU', 'LeakyReLU']
                        type=int, help='Possible activation funcs of the childnet (list of str)')
    parser.add_argument('--verbose', help='Verbose while training the controller/policy (bool)', nargs='?', const=1, 
                        type=bool, default=False)
    parser.add_argument('--num_episodes', help='Nb of episodes the policy net is trained (int)', nargs='?', const=1, 
                        type=int, default=500)
    parser.add_argument('--shared_episodes', help='Nb of episodes the shared net is trained (int)', nargs='?', const=1, 
                        type=int, default=100)
    args = parser.parse_args()
    
    # parameter settings
    args.possible_hidden_units += ['EOS'] # each hidden units end with 'EOS'
    total_actions = args.possible_hidden_units + args.possible_act_functions # total_actions = [1,2,4,8,16,32, 'EOS'] + ['Tanh', 'ReLU']
    n_outputs = len(args.possible_hidden_units) + len(args.possible_act_functions) # output dimension of the PolicyNet
    
    # setup policy network
    policy = Policy(len(args.possible_hidden_units), len(args.possible_act_functions), args.max_layer)
    
    # train
    trainer = Trainer(policy, args.batch, total_actions, args.shared_episodes, args.verbose, args.num_episodes)
    policy = trainer.training()
    
    # save model
    torch.save(policy.state_dict(), 'policy.pt')