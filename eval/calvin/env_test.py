import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"
os.environ["EGL_DEVICE_ID"] = "2"

import pyrender

from calvin_env.envs.play_table_env import get_env

path = "calvin/dataset/calvin_debug_dataset/validation"
env = get_env(path, show_gui=False)
print(env.get_obs())
