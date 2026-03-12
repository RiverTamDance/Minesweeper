from CircularBuffer import CircularBuffer
import CNN
import Generalissimo
import time

# Constants ----------------
memory_capacity = 50


def initialize_replay_buffer(memory_capacity):
   return(CircularBuffer(memory_capacity))


def orchestrator():
   
   D = initialize_replay_buffer(memory_capacity)

   for i in range(50):
      experience = [Generalissimo.main(), i, i/3, Generalissimo.main()]
      D.append(experience)

   return(D)


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
