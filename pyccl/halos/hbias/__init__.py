"""
====================================
Halo Bias (:mod:`pyccl.halos.hbias`)
====================================

Models of the halo bias function.
"""

from ..halo_model_base import HaloBias, halo_bias_from_name
from .bhattacharya11 import *
from .sheth01 import *
from .sheth99 import *
from .tinker10 import *
