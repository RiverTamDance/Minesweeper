from CircularBuffer import CircularBuffer
import CNN
import utils
import time
import random
import numpy as np

# Constants ----------------
BUFFER_SIZE = 1_000_000
C = 10_000
WARM_UP = 50_000
NUMBER_OF_EPISODES = 10_000
MONITOR_INFO = utils.monitor_metadata()



def play_the_game(state, gamestate_generator, train_state):

   action = utils.get_action(state, train_state.Q_policy)
   utils.perform_action(action, MONITOR_INFO)
   gamestate = next(gamestate_generator)

   terminal_state, next_state = utils.get_next_state(gamestate, MONITOR_INFO)


   #here I check if the click did anything. I want clicks that do nothing to have no reward.
   if state == next_state:
      gamestate = "no_change"

   reward = utils.get_reward(gamestate)

   return((state, action, reward, next_state), terminal_state)


def orchestrator():

   train_state = CNN.setup()
   D = CircularBuffer(BUFFER_SIZE)
   gamestate_generator = utils.listen_for_gamestate()

   # episodes loop
   i = 0
   for _ in range(NUMBER_OF_EPISODES):

      utils.restart_game()
      state = utils.get_state(MONITOR_INFO)

      # play the episode
      terminal_state = False
      while not terminal_state:
         
         i += 1
         experience, terminal_state = play_the_game(state, gamestate_generator,train_state)

         D.append(experience)

         if len(D) >= WARM_UP:
            experience_batch = D.rsample(32)
            train(state, experience_batch)

            if i == C:
               i = 0
               target_network.load_state_dict(policy_network.state_dict())
            
         state = next_state



if __name__ == "__main__":

   start_time = time.perf_counter()

   D = orchestrator()
   experiences = D.rsample(32)
   CNN.train(experiences)



   end_time = time.perf_counter()
   print("--- %s seconds ---" % (end_time - start_time))


   # D = orchestrator()
   # for i in range(12):
   #    D.append(i)
   #    print(D)
   # print(D.rsample(3))
   


   # for i in range(50):
   #    action = (random.randint(0,8),random.randint(0,8))
   #    experience = [Generalissimo.main(), action, i/3, Generalissimo.main()]
   #    D.append(experience)
