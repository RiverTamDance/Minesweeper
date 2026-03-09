from collections import namedtuple
import torch.nn as nn
import torch
import torch.nn.functional as F

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
# dtype = torch.float
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")
# torch.set_default_device(device)
# torch.manual_seed(42)


# # Should I initialize the weights of the neural networks?
# # No, PyTorch does it automatically when you call NeuralNetwork(). 
# # Each layer type has its own default initialization scheme — for nn.Linear and nn.Conv2d, 
# # it uses Kaiming uniform initialization (also called He initialization), 
# # which is designed to work well with ReLU activations. It scales the random weights based 
# # on the layer's fan-in so that activations don't explode or vanish as they propagate through the neural network.
# Q_policy = NeuralNetwork().to(device)
# Q_target = NeuralNetwork().to(device)
# Q_target.load_state_dict(Q_policy.state_dict())

# learning_rate = 1e-4
# optimizer = torch.optim.Adam(Q_policy.parameters(), lr=learning_rate)

# Experiences = namedtuple('Experiences', ['states', 'actions', 'rewards', 'next_state'])


# def train(batch):
#    #I might as well get used to the ideas of tensors, as I'm living in a tensor world.
#    # Turn batch into tuples of (state), (action), (reward), (next_state)
#    data = zip(*batch)
#    #y = r if next_state is None else r + gamma*max(Q_target)
#    print(list(data))

# if __name__ == "__main__":
#    batch = [('s1','t',1,'s2'), ('s2','a',-10,'s4')]
#    train(batch)


if __name__ == "__main__":
   torch.manual_seed(123)
   model = NeuralNetwork()
   X = torch.rand((1, 1, 216, 216))
   out = model(X)
   print(out)