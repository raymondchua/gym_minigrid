# Import the envs module so that envs register themselves
from .envs import *

# Import wrappers so it's accessible when installing with pip
from .wrappers import *
from .register import *
from .rendering import *
from .roomgrid import *
from .window import *