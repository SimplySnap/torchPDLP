def presolve(input_file, reduced_file, postsolve_data, verbose=False):
  '''
  Takes an input mps file, presolves it using the papilo algorithm, and outputs
  a reduced mps file and the data that will be needed to postsolve the solution.

  Args:
    input_file: path to the original mps file
    reduced_file: path to output the reduced mps file
    postsolve_data: path to output the data file for postsolve

  Returns nothing, but outputs the reduced mps and postsolve data to designated paths.

  '''
  import subprocess as sb

  try:
      # Construct the command to run Papilo's presolve command
      command = [
          "papilo", "presolve",         # The executable name (now in PATH)
          "-f", input_file,
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

