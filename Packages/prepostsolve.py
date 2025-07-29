def presolve(original_file, reduced_file, postsolve_data, verbose=False):
  '''
  Takes an input mps file, presolves it using the papilo algorithm, and outputs
  a reduced mps file and the data that will be needed to postsolve the solution.

  Args:
    original_file: path to the original mps file
    reduced_file: path to output the reduced mps file
    postsolve_data: path to output the data file for postsolve

  Returns nothing, but outputs the reduced mps and postsolve data to designated paths.

  '''
  import subprocess as sb

  try:
      # Construct the command to run Papilo's presolve command
      command = [
          "papilo", "presolve",         # The executable name (now in PATH)
          "-f", original_file,
          "-r", reduced_file,
          "-v", postsolve_data
      ]

      # Run the command
      result = sb.run(command, capture_output=True, text=True, check=True)

      print("Presolving with", result.stdout) if verbose else None

      # Now you can work with the reduced_output_file in your Python code

  except FileNotFoundError:
      print(f"Error: 'papilo' executable not found. Make sure it's in your PATH or specify the full path.")
  except sb.CalledProcessError as e:
      print(f"Error running Papilo: {e}")
      print("Stdout:", e.stdout)
      print("Stderr:", e.stderr)
  except Exception as e:
      print(f"An unexpected error occurred: {e}")
  except Exception as e:
      print(f"An unexpected error occurred: {e}")


def postsolve(minimizer, postsolve_data, reduced_mps, output_path="postsolve_solution.sol"):
    '''
    Postsolves the minimizer solution

    Args:
      minimizer: torch tensor minimizer solution to the reduced LP
      postsolve_data: path to postsolve data from presolve
      reduced_mps: the mps file containing the presolved LP
      output_path: path to output the postsolve solution

    Returns nothing, but outputs the solution to the original LP to the output_path.
    '''
    import torch
    import subprocess

    minimizer = minimizer.flatten()

    # Step 1: extract variable names from reduced LP
    """
    Parse variable names from the COLUMNS section of an MPS file.
    Returns a list of variable names in the order they appear.
    """

    var_names = []
    seen = set()
    in_columns = False

    with open(reduced_mps, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("COLUMNS"):
                in_columns = True
                continue
            if line.startswith("RHS") or line.startswith("BOUNDS") or line.startswith("ENDATA"):
                break
            if in_columns:
                tokens = line.split()
                if len(tokens) >= 2:
                    var = tokens[0]
                    if var not in seen:
                        var_names.append(var)
                        seen.add(var)

    # Step 2: write reduced.sol
    """
    Writes the reduced solution to a .sol file, line-by-line:
    <var name> <value>
    """
    with open("reduced.sol", 'w') as f:
        for var, val in zip(var_names, minimizer.tolist()):
            f.write(f"{var} {val:.16f}\n")

    # Step 3: call PaPILO postsolve
    subprocess.run([
        "papilo", "postsolve",
        "-v", postsolve_data,
        "-u", "reduced.sol",
        "-l", output_path
    ])
    return
