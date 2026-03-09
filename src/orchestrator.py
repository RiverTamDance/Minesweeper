from CircularBuffer import CircularBuffer

# Constants ----------------
memory_capacity = 1_000_000


def initialize_replay_buffer(memory_capacity):
   return(CircularBuffer(memory_capacity))


def orchestrator():
   
   D = initialize_replay_buffer(memory_capacity)
   return(D)


# if __name__ == "__main__":







   # D = orchestrator()
   # for i in range(10):
   #    D.append(i)
   #    print(D)
