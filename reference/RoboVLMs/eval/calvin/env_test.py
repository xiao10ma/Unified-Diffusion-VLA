import os

# BASE_DIR is the path to the RoboVLMs directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "/data/user/wsong890/user68/data/task_ABCD_D/validation")

from calvin_env.envs.play_table_env import get_env
env = get_env(path, show_gui=False)
print(env.get_obs())
