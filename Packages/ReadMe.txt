To run these scripts from a notebook, clone the repository locally by
          !rm -rf PDLP-AMD-RIPS
          !git clone --branch main https://github.com/SimplySnap/PDLP-AMD-RIPS.git
          !cd PDLP-AMD-RIPS

Change main to whatever branch you'd like to clone

To run a script, pull from the cloned directory by
          %run "PDLP-AMD-RIPS/.../script.py"

You can also run from a github codespace notebook without cloning, just remove the PDLP-AMD-RIPS in the file path.
However, this will not work to define functions, at least it didn't work for me in collab, but you can run
          %run PDLP-AMD-RIPS/Packages/setup.py
and then you can pull functions from modules like this
          from pdhg_torch_algorithm import pdhg_torch
Just using import pdhg_torch_algorithm does not work for some reason.

Make sure any modules you call from other functions are imported in those functions or they won't work, this incudes modules like torch, numpy, and cplex.
To minimize the time it takes to import larger libraries like torch, run
          %run PDLP-AMD-RIPS/Packages/libraries.py
initially.
          
