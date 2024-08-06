import pendulum

def get_current_date():
    return pendulum.now().strftime('%Y%m%d_%H%m%S')

def add_prefix_to_keys(dict_, prefix):
  """Adds a prefix to all keys in a dictionary.

  Args:
    dict_: The input dictionary.
    prefix: The prefix to add.

  Returns:
    A new dictionary with prefixed keys.
  """

  return {prefix + key: value for key, value in dict_.items()}