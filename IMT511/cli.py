import os
print(os.getcwd())

from pathlib import Path

current_folder = Path.cwd().name
print(current_folder)
