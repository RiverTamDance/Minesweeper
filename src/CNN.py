from dataclasses import dataclass
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import utils


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

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
dtype = torch.float
torch.set_default_device(device)
torch.manual_seed(42)
#torch.set_printoptions(profile="full")

# # Should I initialize the weights of the neural networks?
# # No, PyTorch does it automatically when you call NeuralNetwork(). 
# # Each layer type has its own default initialization scheme — for nn.Linear and nn.Conv2d, 
# # it uses Kaiming uniform initialization (also called He initialization), 
# # which is designed to work well with ReLU activations. It scales the random weights based 
# # on the layer's fan-in so that activations don't explode or vanish as they propagate through the neural network.

# Experiences = namedtuple('Experiences', ['states', 'actions', 'rewards', 'next_state'])


def train(train_state, experiences):
   #I might as well get used to the ideas of tensors, as I'm living in a tensor world.
   # Turn batch into tuples of (state), (action), (reward), (next_state)

   batch = utils.SARS(*zip(*experiences))

   #Time for a little dialogue on dimensions
   # state (216x216) -> states (32x1x216x216)
   # action (2) ->  actions (32x2) -> (32x1)
   # reward (1) ->  rewards (32x1)
   # next_state (216x216) -> next_states (32x216x216)

   # gotta use np.array() to create a master array of all the sub arrays, for performance reasons.
   states = torch.tensor(np.array(batch.state), dtype=torch.float, device=device).unsqueeze(1)


   #actions are pairs of 0-based cell coordinates from (0,0) to (8,8), and we remap them to a number from 0 to 80
   actions = [x*9+y for x,y in batch.action]
   actions = torch.tensor(actions, device=device).unsqueeze(1)
   rewards = torch.tensor(batch.reward, device=device).unsqueeze(1)
   next_states = torch.tensor(np.array(batch.state),dtype=torch.float, device=device).unsqueeze(1)

   policy_values = train_state.Q_policy(states)

   # The output of a single Q_policy run is an 81 dim tensor, and over all states its a 32x81 vector.
   # To meaningfully do gradient descent, I want to find the MSE(y-Q(s,a)), so I need to get the value 
   # corresponding to the action in our experience history. I want one element per row.
   # Gather appears to be the correct method for this operation. Claude has been insistant on that.
   # we want to gather according to column, naturally. There is one row per experience (32 rows in a mini-batch),
   # and there are 81 actions per row. We want to gather 1 action per row, which means we gather along the column axis.
   policy_values = policy_values.gather(dim=1, index=actions)

   # I have changed things around, and now a next_state is terminal if it is all black.
   game_over = (next_states == 0).all()
   y = torch.where(game_over, rewards, rewards+train_state.gamma*torch.max(train_state.Q_target(next_states))) 

   #time for actual training!

   return(None)

@dataclass
class TrainState:
   gamma: float
   learning_rate: float
   Q_policy: nn.Module
   Q_target: nn.Module
   optimizer: torch.optim.Optimizer


def setup(weights_path) -> TrainState:

   gamma = 0.99
   learning_rate = 1e-4
   Q_policy = NeuralNetwork().to(device)
   Q_policy.load_state_dict(torch.load(weights_path, weights_only=True))
   Q_target = NeuralNetwork().to(device)
   Q_target.load_state_dict(Q_policy.state_dict())
   optimizer = torch.optim.Adam(Q_policy.parameters(), lr=learning_rate)

   return TrainState(
      gamma = gamma,
      learning_rate = learning_rate,
      Q_policy = Q_policy,
      Q_target = Q_target,
      optimizer = optimizer
   )

   

if __name__ == "__main__":
   print("don't run this file directly")