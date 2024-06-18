import re
import os
import numpy as np
from time import time, sleep

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = str(value)
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)
    return value

def corrcoef(dist_a, dist_b):
    """Returns a scalar between 1.0 and -1.0. 0.0 is no correlation. 1.0 is perfect correlation"""
    dist_a = np.copy(dist_a)  # Prevent np.corrcoef from blowing up on data with 0 variance
    dist_b = np.copy(dist_b)
    dist_a[0] += 1e-12
    dist_b[0] += 1e-12
    return np.corrcoef(dist_a, dist_b)[0, 1]

def make_run_log_dirs(save_dir, env_id, independent, rp_mode, real_rewards, name, n_agents):
    predictors = {0: 'ComparisonRewardPredictor',
                 1: 'PrmComparisonRewardPredictor',
                 2: 'CrmComparisonRewardPredictor'}
    log_dir = os.path.join(save_dir,
                               f"{env_id}{'-independent' if independent else ''}{'-' + predictors[rp_mode]}{'-RealRewards' if real_rewards else ''}",
                               f"{name}-{int(time())}")

    return [os.path.join(log_dir, f"agent_{i}") for i in range(n_agents)], log_dir
         
     

    
