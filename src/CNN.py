from collections import namedtuple
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
   def __init__(self):
      # See the excel worksheet for a breakdown of where these numbers come from. 
      # They are modified Mnih 2015, with help from Claude to make the modifications. 
      # The shape of the input to a Conv2d object is (batch count, channels, height, width)
      # or (channels, height, width). In our case, either (1, 216, 216) or (1,1,216,216) works.

      super().__init__()

      # This requires an input shape of (N, 1, 216, 216), where N is batch count 
      self.layers = nn.Sequential(
         nn.Conv2d(1, 32, 8, 4),
         nn.ReLU(),
         nn.Conv2d(32, 64, 5, 2),
         nn.ReLU(),
         nn.Conv2d(64, 64, 3, 2),
         nn.ReLU(),
         nn.Conv2d(64, 64, 3, 1),
         nn.ReLU(),
         # If I have a batch count other than one, this seems to need start_dim=1. 
         # If the batch count is 1, then this should be 0
         # This means that we flatten along dimensions (channels, height, width) that the layers were designed around.
         nn.Flatten(start_dim=1),
         nn.Linear(6400, 512),
         nn.ReLU(),
         nn.Linear(512, 81)
      )

   def forward(self, x):
      output = self.layers(x)
      return output

      # This is the trick typically used to determine the size of the input for the first fully connected layer.
      # dummy = torch.zeros(1,216, 216)
      # x = self.conv3(self.conv2(self.conv1(dummy)))
      # flat_size = x.view(1, -1).shape[1]
      # print(flat_size)

# # setup
dtype = torch.float
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
torch.set_default_device(device)
torch.manual_seed(42)


# # Should I initialize the weights of the neural networks?
# # No, PyTorch does it automatically when you call NeuralNetwork(). 
# # Each layer type has its own default initialization scheme — for nn.Linear and nn.Conv2d, 
# # it uses Kaiming uniform initialization (also called He initialization), 
# # which is designed to work well with ReLU activations. It scales the random weights based 
# # on the layer's fan-in so that activations don't explode or vanish as they propagate through the neural network.


# learning_rate = 1e-4
# optimizer = torch.optim.Adam(Q_policy.parameters(), lr=learning_rate)

# Experiences = namedtuple('Experiences', ['states', 'actions', 'rewards', 'next_state'])


def train(experiences):
   #I might as well get used to the ideas of tensors, as I'm living in a tensor world.
   # Turn batch into tuples of (state), (action), (reward), (next_state)

   gamma = 0.99

   Q_policy = NeuralNetwork().to(device)
   Q_target = NeuralNetwork().to(device)
   Q_target.load_state_dict(Q_policy.state_dict())

   SARS = namedtuple("SARS", "states actions rewards next_states")
   batch = SARS(*zip(*experiences))

   states = torch.tensor(np.array(batch.states), dtype=torch.float, device=device).unsqueeze(1)
   print(f"states shape: {states.shape}")
   actions = torch.tensor(batch.actions, device=device)
   rewards = torch.tensor(batch.rewards, device=device)
   next_states = torch.tensor(np.array(batch.states), device=device)

   target_values = Q_target.forward(states)

   max_values = torch.max(target_values)
   print(f"max_values and its shape: {max_values}, {max_values.shape}")
   print(max_values)
   
   # The playing field seems to only contain true blacks when the game is over either through a mine
   # appearing, or through the cells containing flags.
   game_over = (next_states == 0).any()
   print(game_over)
   y = torch.where(game_over, rewards, rewards+gamma*max_values) 
   print(rewards)
   print(y)

   # print("it worked")

   #TODO: write an equation for y.

   #Time for a little dialogue on dimensions
   # state (216x216) -> states (32x216x216)
   # action (1) ->  actions (32x1)
   # reward (1) ->  rewards (32x1)
   # next_state (216x216) -> next_states (32x216x216)

#    data = zip(*batch)
#    #y = r if next_state is None else r + gamma*max(Q_target)
#    print(list(data))

# if __name__ == "__main__":
#    batch = [('s1','t',1,'s2'), ('s2','a',-10,'s4')]
#    train(batch)


if __name__ == "__main__":
   print("don't run this file directly")
   # batch = [[1,2,3,4], [2,3,4,5], ['t','s','u','v']]
   # train(batch)


   # torch.manual_seed(123)
   # model = NeuralNetwork()
   # X = torch.rand((1, 1, 216, 216))
   # out = model(X)

   # # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
   # # print(f"Using {device} device")
   # # torch.set_default_device(device)
   # x = torch.arange(24)
   # x = torch.reshape(x, [3,2,4])
   # # x = torch.unsqueeze(x, -2)
   # print(x)
   # print(x.shape)
   # # x[pillar,row,column]
   # print(x[1,0,:]) #Returns all columns of the first row (of two) found at the second depth (of three).
   # print(x[:,0,0]) #Returns the pillar found at the first row and column 
   # print(x[1,:,:]) #Returns both rows found at depth 2
