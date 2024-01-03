#1、导入numpy库并简写为 np
from numpy.lib import stride_tricks
import pandas as pd
import scipy.spatial as quick
import os
from numpy.lib.function_base import meshgrid
from numpy.core.function_base import linspace
import numpy as np

#2、打印numpy的版本和配置说明
print(np.__version__)
print(np.show_config())