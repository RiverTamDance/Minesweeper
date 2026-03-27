class Agent:

   def __init__(self):
      pass

   

   def epsilon(episode_count: int) -> float:
      epsilon = (EPSILON_UPPER_BOUND-episode_count)/EPSILON_UPPER_BOUND
      return epsilon

   def get_action(state: NDArray, policy_network: nn.Module, episode_count) -> tuple[int,int]:

      epsilon = max(get_epsilon(episode_count), EPSILON_MINIMUM)

      if random.random() < epsilon:
         action = (random.randint(0,8), random.randint(0,8))
      else:
         tensor_state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
         tensor_state = tensor_state.unsqueeze(0)
         action = int(torch.argmax(policy_network(tensor_state)).item())
         action = (a := action//9, action-a*9)
      return(action)