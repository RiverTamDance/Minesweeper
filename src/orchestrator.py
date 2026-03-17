from numpy.typing import NDArray
from CircularBuffer import CircularBuffer
import CNN
import utils
import time
import random
import numpy as np
from pathlib import Path
import os

# Constants ----------------
BUFFER_SIZE = 1_000_000
C = 10_000
WARM_UP = 5_000
NUMBER_OF_EPISODES = 20_000
MONITOR_INFO = utils.monitor_metadata()

# Paths -------------------
project_path = Path(__file__).parent.parent
game_file = project_path / "m.love"
assert os.path.isfile(game_file)


def play_the_game(state: NDArray, gamestate_generator, train_state):

   action = utils.get_action(state, train_state.Q_policy)
   utils.perform_action(action, MONITOR_INFO)
   gamestate = next(gamestate_generator)

   terminal_state, next_state = utils.get_next_state(gamestate, MONITOR_INFO)


   #here I check if the click did anything. I want clicks that do nothing to have no reward.
   if (state == next_state).all():
      gamestate = "no_change"

   reward = utils.get_reward(gamestate)

   return((state, action, reward, next_state), terminal_state, gamestate)


def orchestrator():

   train_state = CNN.setup()
   D = CircularBuffer(BUFFER_SIZE)
   gamestate_generator = utils.listen_for_gamestate()
   os.startfile(game_file)
   time.sleep(4)

   # episodes loop
   i = 0
   for episode in range(NUMBER_OF_EPISODES):
      print(f"episode: {episode}")

      if episode >= 1:
         utils.restart_game()
      
      state = utils.get_state(MONITOR_INFO)

      # play the episode
      turn = 0
      terminal_state = False
      while not terminal_state:
         
         
         experience, terminal_state, gamestate = play_the_game(state, gamestate_generator,train_state)
         print(f"turn: {turn} '{gamestate}', reward: {experience[2]}")
         D.append(experience)

         if len(D) >= WARM_UP:
            experience_batch = D.rsample(32)
            CNN.train(train_state, experience_batch)

            if i == C:
               i = 0
               train_state.Q_target.load_state_dict(train_state.Q_policy.state_dict())
         
         next_state = experience[-1]
         state = next_state
         turn+=1



# if __name__ == "__main__":

#    start_time = time.perf_counter()

#    D = orchestrator()
#    experiences = D.rsample(32)
#    CNN.train(experiences)



#    end_time = time.perf_counter()
#    print("--- %s seconds ---" % (end_time - start_time))


   # D = orchestrator()
   # for i in range(12):
   #    D.append(i)
   #    print(D)
   # print(D.rsample(3))
   


   # for i in range(50):
   #    action = (random.randint(0,8),random.randint(0,8))
   #    experience = [Generalissimo.main(), action, i/3, Generalissimo.main()]
   #    D.append(experience)
