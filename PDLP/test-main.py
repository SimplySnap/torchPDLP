import sys
import runpy

# Set sys.argv as if you were running from the command line
sys.argv = [
    "main.py",  # the script name (can be anything)
    "--instance_path", "experiments/datasets",
    "--tolerance", "1e-4",
    "--precondition",
    "--fishnet",
    "--max_kkt", "1000",
    "--time_limit", "600",
    "--verbose"
    # Add or remove arguments as needed
]

# Run main.py as a script
runpy.run_path("PDLP/main.py")