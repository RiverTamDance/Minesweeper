from mss.screenshot import ScreenShot
import pyautogui
import mss
import numpy as np
from numpy.typing import NDArray
import cv2
from CircularBuffer import CircularBuffer
import random
import torch
import socket
import torch.nn as nn
import utils
import time
from collections import namedtuple

# constants
CELL_WIDTH = CELL_HEIGHT = 24
EPSILON_UPPER_BOUND = 50_000
EPSILON_MINIMUM = 0.1
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def initialize_replay_buffer(memory_capacity):
   return(CircularBuffer(memory_capacity))

SARS = namedtuple("SARS", "state action reward next_state")

def monitor_metadata() -> dict[str, int]:
   """setup some monitor metadata used throughout the program"""

   # constants
   MONITOR_NUMBER = 1
   PIXELS_FROM_TOP = 1039
   PIXELS_FROM_LEFT = 1811
   HEIGHT = WIDTH = 216

   with mss.mss() as sct:
      # Part of the screen to capture
      monitor_defaults: dict[str, int] = sct.monitors[MONITOR_NUMBER]
      monitor_info: dict[str, int] = {
         'top': monitor_defaults['top'] + PIXELS_FROM_TOP,
         'left': monitor_defaults['left'] + PIXELS_FROM_LEFT,
         'width': WIDTH,
         'height': HEIGHT,
         'mon': MONITOR_NUMBER,
      }

   return monitor_info
   

def get_raw_state(monitor_info) -> ScreenShot:
   """returns a screenshot from mss"""

   with mss.mss() as sct:
      screenshot: ScreenShot = sct.grab(monitor_info)

   return screenshot


def preprocess_state(screenshot: ScreenShot) -> NDArray:
   """prepocess the screenshot by converting to gray values"""

   img: NDArray = np.array(screenshot)  # noqa: F821
   state: NDArray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
   return state


def get_state(monitor_info)-> NDArray:
   """ compose raw state and preprocess state """

   screenshot = get_raw_state(monitor_info)
   state = preprocess_state(screenshot)
   return(state)


def listen_for_gamestate(port=12345):
   srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   srv.bind(("127.0.0.1", port))
   srv.listen(1)
   print("Waiting for connection...")
   conn, _ = srv.accept()
   print("Connected!")
   conn.setblocking(False)
   buffer = ""
   while True:
      try:
         data = conn.recv(4096).decode()
         if not data:
               return
         buffer += data
      except BlockingIOError:
         pass
      
      if "\n" in buffer:
         lines = buffer.split("\n")
         buffer = lines[-1]  # keep incomplete trailing data
         yield lines[-2]     # yield the last complete line


def get_next_state(gamestate, monitor_info) -> tuple[bool, NDArray]:
   if gamestate in ["new_game", "playing", "no_change"]:
      terminal_state = False
      next_state = utils.get_state(monitor_info)
   elif gamestate in ["game_over", "victory"]:
      terminal_state = True
      next_state = np.zeros((216, 216), dtype=np.float32)
   else:
      raise Exception(f"unanticipated gamestate {gamestate}")
   
   return(terminal_state, next_state)

def get_epsilon(episode_count: int) -> float:
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


def perform_action(action: tuple[int, int], monitor_info: dict[str, int]) -> None:
   """click on one of the tiles using pyautogui"""

   col, row = action

   # determine on-screen position of the cell that will be clicked
   # These are 0-based cell coordinates.
   cell_x: int = monitor_info['left'] + ((col) * CELL_WIDTH) + CELL_WIDTH // 2
   cell_y: int = monitor_info['top'] + ((row) * CELL_HEIGHT) + CELL_HEIGHT // 2

   # Execute action
   pyautogui.click(cell_x, cell_y)

   return None

def restart_game():
   """reset the game board for a new episode"""
   window = pyautogui.getWindowsWithTitle("XP Minesweeper Classic")[0]
   window.activate()
   time.sleep(0.1)
   pyautogui.click(1922, 979)
   return None

def save_weights(state_dict, file_path):
   torch.save(state_dict, file_path)
   return None

def get_reward(gamestate):
   if gamestate in ["new_game", "playing"]:
      reward = 0.1
   elif gamestate == "no_change":
      reward = 0
   elif gamestate == "victory":
      reward = 1
   elif gamestate == "game_over":
      reward = -1
   else:
      raise Exception(f"unanticipated gamestate {gamestate}")
   return(reward)


def main():
   monitor_info: dict[str, int] = monitor_metadata()

   screenshot: ScreenShot = get_raw_state(monitor_info)
   state: NDArray[np.uint8] = preprocess_state(screenshot)

   #action: tuple[int, int] = test_action
   #perform_action(action, monitor_info)
   return(state)


if __name__ == '__main__':
   print(pyautogui.getAllTitles())


# -------- Old code -------------------------------------------------