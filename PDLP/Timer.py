from time import perf_counter

class Timer:
  """
  Timer class to measure execution time of code blocks.
  Usage:
  
    with Timer("Label"):
        # Code block to be timed

  Output:
    Label: <time in seconds> seconds
  """
  def __init__(self, label="Elapsed time"):
      self.label = label

  def __enter__(self):
      self.start = perf_counter()
      return self

  def __exit__(self, *args):
      self.end = perf_counter()
      self.elapsed = self.end - self.start
      print(f"{self.label}: {self.elapsed:.6f} seconds")
