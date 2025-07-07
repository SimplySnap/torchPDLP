import sys
import os

# Get the current working directory
current_dir = os.getcwd()

# Add the 'Packages' directory within the cloned repository to the system path
# Assuming the repository was cloned into a directory named 'PDLP-AMD-RIPS'
repo_dir = os.path.join(current_dir, 'PDLP-AMD-RIPS', 'Packages')
sys.path.append(repo_dir)

# Now you can import files from the 'Packages' directory.
# Replace 'your_module' with the actual name of the .py file you want to import (without the .py extension).
# For example, if you want to import 'pdhg_solver.py', you would use 'import pdhg_solver'
# try:
#     import your_module
#     print("Successfully imported your_module")
# except ImportError as e:
#     print(f"Error importing module: {e}")

print(f"'{repo_dir}' has been added to the system path.")
print("You can now import .py files from this directory.")
