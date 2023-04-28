from pymoo.indicators.hv import HV
import numpy as np


pareto = [(0.3,0.8), (0.5,0.5),(0.7,0.4)]

pareto = np.array(pareto)

hv = HV(ref_point=(0, 0))
hv_value = hv.do(pareto)


print(hv_value)