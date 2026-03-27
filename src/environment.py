import pyautogui
import numpy as np
from numpy.typing import NDArray
from mss.screenshot import ScreenShot
import mss
import cv2
pyautogui.PAUSE = 0.0
CELL_WIDTH = CELL_HEIGHT = 24

class Environment:

   def setup(self):
      pass


   def step(self, action):
      pass

   def __init__(self):
      self.game_window_coords = game_window_coords()

   def game_window_coords() -> dict[str, int]:
      """setup some monitor metadata used throughout the program"""

      # constants
      MONITOR_NUMBER = 1
      PIXELS_FROM_TOP = 1039
      PIXELS_FROM_LEFT = 1811
      HEIGHT = WIDTH = 216

      with mss.mss() as sct:
         # Part of the screen to capture
         monitor_defaults: dict[str, int] = sct.monitors[MONITOR_NUMBER]
         game_window_coords: dict[str, int] = {
            'top': monitor_defaults['top'] + PIXELS_FROM_TOP,
            'left': monitor_defaults['left'] + PIXELS_FROM_LEFT,
            'width': WIDTH,
            'height': HEIGHT,
            'mon': MONITOR_NUMBER,
         }

      return game_window_coords

   def _get_raw_state(self, game_window_coords) -> ScreenShot:
      """returns a screenshot from mss"""

      with mss.mss() as sct:
         screenshot: ScreenShot = sct.grab(game_window_coords)

      return screenshot

   def _get_preprocessed_state(self, game_window_coords) -> NDArray:
      """prepocess the screenshot by converting to gray values"""

      screenshot = self._get_raw_state(game_window_coords)

      img: NDArray = np.array(screenshot)  # noqa: F821
      state: NDArray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
      return state

   def get_state(gamestate, game_window_coords) -> tuple[bool, NDArray]:


      if gamestate in ["new_game", "playing", "no_change"]:
         terminal_state_flag = False
         state = self._get_preprocessed_state(game_window_coords)
      elif gamestate in ["game_over", "victory"]:
         terminal_state_flag = True
         state = np.zeros((216, 216), dtype=np.float32)
      else:
         raise Exception(f"unanticipated gamestate {gamestate}")
      
      return(terminal_state_flag, next_state)



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

   def restart_game(self):
      """reset the game board for a new episode"""
      window = pyautogui.getWindowsWithTitle("XP Minesweeper Classic")[0]
      window.activate()
      time.sleep(0.1)
      pyautogui.click(1922, 979)
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

   def perform_action(action: tuple[int, int], game_window_coords: dict[str, int]) -> None:
   """click on one of the tiles using pyautogui"""

   col, row = action

   # determine on-screen position of the cell that will be clicked
   # These are 0-based cell coordinates.
   cell_x: int = game_window_coords['left'] + ((col) * CELL_WIDTH) + CELL_WIDTH // 2
   cell_y: int = game_window_coords['top'] + ((row) * CELL_HEIGHT) + CELL_HEIGHT // 2

   # Execute action
   pyautogui.click(cell_x, cell_y)

   return None