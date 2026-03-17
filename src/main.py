import orchestrator
import time

def main():
   start_time = time.perf_counter()
   orchestrator.orchestrator()
   print("--- %s seconds ---" % (time.perf_counter() - start_time))

if __name__ == "__main__":
   main()