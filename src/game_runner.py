# try to run the love file.
import subprocess
import os
from pathlib import Path

project_path = Path(__file__).parent.parent
game_file = project_path / "V11.love"

assert os.path.isfile(game_file)

os.startfile(game_file)