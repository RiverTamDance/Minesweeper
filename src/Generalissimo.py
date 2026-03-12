from mss.screenshot import ScreenShot
import pyautogui
import mss
import numpy as np
from numpy.typing import NDArray
import cv2
from CircularBuffer import CircularBuffer

# constants

test_action = (9, 9)
CELL_WIDTH = CELL_HEIGHT = 24


def initialize_replay_buffer(memory_capacity):
   return(CircularBuffer(memory_capacity))


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


def perform_action(action: tuple[int, int], monitor_info: dict[str, int]) -> None:
   """click on one of the tiles using pyautogui"""

   col, row = action

   # determine on-screen position of the cell that will be clicked
   # These are 1-based cell coordinates.
   cell_x: int = monitor_info['left'] + ((col - 1) * CELL_WIDTH) + CELL_WIDTH // 2
   cell_y: int = monitor_info['top'] + ((row - 1) * CELL_HEIGHT) + CELL_HEIGHT // 2

   # Execute action
   pyautogui.click(cell_x, cell_y)

   return None


def main():
   monitor_info: dict[str, int] = monitor_metadata()

   screenshot: ScreenShot = get_raw_state(monitor_info)
   state: NDArray[np.uint8] = preprocess_state(screenshot)

   #action: tuple[int, int] = test_action
   #perform_action(action, monitor_info)
   return(state)


if __name__ == '__main__':
   s = main()
   for r in s:
      print(r)


# -------- Old code -------------------------------------------------

# Display the picture
# cv2.imshow('test1_colour', img)
# cv2.imshow('test1_gray', gray)

# # I'm not sure this code is necessary if I'm not actually opening a cv2 window.
# # Press "q" to quit
# if cv2.waitKey(0) & 0xFF == ord('q'):
#    cv2.destroyAllWindows()
