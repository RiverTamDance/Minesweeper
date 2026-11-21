import random


class CircularBuffer:
   def __init__(self, max_len):
      self.max_len = max_len
      self.buffer = [None] * max_len
      # oldest_index will always be one ahead of the most recently appended value.
      self.oldest_index = 0
      self.size = 0

   def __getitem__(self, index):
      return self.buffer[index]

   def __len__(self):
      return self.size

   def append(self, element):
      self.buffer[self.oldest_index] = element
      self.oldest_index += 1
      self.oldest_index %= self.max_len
      # Increment size until it is the size of max_len
      self.size = min(self.size + 1, self.max_len)

   def rsample(self, k):
      if k > self.size:
         raise ValueError(f'Sample size {k} is too big for a buffer with {self.size} elements')
      elif self.size == self.max_len:
         return random.sample(self.buffer, k)
      else:
         return random.sample(self.buffer[: self.oldest_index], k)

   def __repr__(self):
      return f'CircularBuffer({self.max_len})'

   def __str__(self):
         return f'{self.buffer[:10]}'
