# Import the envs module so that envs register themselves
# from . import envs
import CRLMSF.environments.gym_minigrid.envs

# Import wrappers so it's accessible when installing with pip
import CRLMSF.environments.gym_minigrid.wrappers