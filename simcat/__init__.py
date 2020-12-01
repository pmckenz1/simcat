#!/usr/bin/env

# imports

from .Database import Database    # BUILDS THE DATABASE OF LABELS
from .Analysis import Analysis          # POST-SIM ANALYSIS
from .Simulator import Simulator
from . import plot                #
# from .Streamer import Streamer  # PULLS FROM DB TO RUN SIMS ON ENGINES
# from .Model import Model        # DEPRECATED TO SIMCOAL
# from .Genes import Genes        # DEPRECATED TO SIMCOAL
# from . import utils             #



# dunders
__version__ = "0.0.6"
__authors__ = "Patrick McKenzie and Deren Eaton"
