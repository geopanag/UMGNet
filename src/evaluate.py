import numpy as np

import pandas as pd


def uplift_score(
    prediction: np.ndarray, 
    treatment: np.ndarray, target: np.ndarray, rate: float = 0.2
) -> float:
    """
    From https://ods.ai/competitions/x5-retailhero-uplift-modeling/data
    Order the samples by the predicted uplift.
    Calculate the average ground truth outcome of the top rate*100% of the treated and the control samples.
    Subtract the above to get the uplift.
    """
    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()

    control_n = int((treatment == 0).sum() * rate)
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    score = treatment_p - control_p
    return score




