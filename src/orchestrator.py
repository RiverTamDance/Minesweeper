from numpy.typing import NDArray
from CircularBuffer import CircularBuffer
from collections import Counter
import CNN
import utils
import time
import numpy as np
from pathlib import Path
import os
from torch.profiler import profile, record_function, ProfilerActivity

# Constants ----------------
BUFFER_SIZE = 1_000_000
C = 5_000
WARM_UP = 100*10 #estimating 10 obs per ep.
NUMBER_OF_EPISODES = 100_000
MONITOR_INFO = utils.monitor_metadata()
np.set_printoptions(threshold=82)

# Paths -------------------
project_path = Path(__file__).parent.parent
game_file = project_path / "m.love"
log_path = project_path / "log"
tile_count_log_file = log_path / "tile_counts.log"
gamestate_counter_file = log_path / "gamestate_counts.log"
model_weights_path = project_path / "weights"
model_weights_file = model_weights_path / "policy_weights.pt"

assert os.path.isfile(game_file)

def log_tile_visits(visit_counts: NDArray):
   np.savetxt(tile_count_log_file, visit_counts, fmt='%d', delimiter= ' ')
   # with tile_count_log_file.open('w', encoding = "utf-8") as file:
   #    file.write()

def log_gamestates(gamestate_counter):
   with gamestate_counter_file.open("w", encoding="utf-8") as file:
      for key, count in gamestate_counter.items():
         file.write(f"{key}\t{count}\n")


def play_the_game(state: NDArray, gamestate_generator, train_state, episode_count):

   action = utils.get_action(state, train_state.Q_policy, episode_count)
   utils.perform_action(action, MONITOR_INFO)
   gamestate = next(gamestate_generator)
   terminal_state, next_state = utils.get_next_state(gamestate, MONITOR_INFO)

   #here I check if the click did anything. I want clicks that do nothing to have no reward.
   if (state == next_state).all():
      gamestate = "no_change"

   reward = utils.get_reward(gamestate)
   experience = utils.SARS(state, action, reward, next_state)

   return(experience, terminal_state, gamestate)


def orchestrator():

   gamestate_generator = utils.listen_for_gamestate()
   os.startfile(game_file)
   D = CircularBuffer(BUFFER_SIZE)
   train_state = CNN.setup(model_weights_file)
   visit_counts = np.zeros((9, 9), dtype=np.uint64)
   gamestate_counter = Counter()
   time.sleep(1)

   # episodes loop
   turn = 0
   for episode in range(NUMBER_OF_EPISODES):
      print(f"episode: {episode}")

      if episode >= 1:
         utils.restart_game()

      if episode % 100 == 0:
         log_tile_visits(visit_counts)
         print(visit_counts)
         log_gamestates(gamestate_counter)

      if episode % 1000 == 0:
         utils.save_weights(train_state.Q_policy.state_dict(), model_weights_file)

         #using at exit to save the weights is a cool idea in theory, but turn don't think it's worth doing in practice
         # atexit.unregister(utils.save_weights)
         # atexit.register(utils.save_weights, train_state.Q_policy.state_dict(), model_weights_file)
      
      state = utils.get_state(MONITOR_INFO)

      # play the episode
      turn = 0
      terminal_state = False
      while not terminal_state:
         
         experience, terminal_state, gamestate = play_the_game(state, gamestate_generator,train_state, episode)
         D.append(experience)

         if len(D) >= WARM_UP:
            experience_batch = D.rsample(32)

            with profile(
               activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
               record_shapes=True,
               profile_memory=True,
            ) as prof:
               CNN.train(train_state, experience_batch)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

            if turn % C == 0:
               train_state.Q_target.load_state_dict(train_state.Q_policy.state_dict())
         
         next_state = experience.next_state
         state = next_state
         turn+=1
         visit_counts[experience.action] += 1
         gamestate_counter[gamestate] += 1



# if __name__ == "__main__":