import random
import numpy as np

def make_task(len_node):
  tasks = list(range(0, len_node))
  random.shuffle(tasks)
  tasks_2d = [[task] for task in tasks]
  return tasks_2d

if __name__ == "__main__":
  print("They are tasks :", make_task(10))
  dest_state = np.zeros(5, dtype=int)
  dest_state[1] = 1
  print(dest_state)
  