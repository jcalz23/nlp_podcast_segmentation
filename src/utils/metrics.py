import numpy as np
from nltk.metrics.segmentation import windowdiff

def windiff(y_true, y_pred, window_size):
    """
    Calculate the WindowDiff metric for segmentation evaluation using NLTK.

    Args:
        y_true (np.array): Ground truth segmentation.
        y_pred (np.array): Predicted segmentation.
        window_size (int): Size of the comparison window.

    Returns:
        float: WindowDiff score (lower is better).
    """
    # Convert numpy arrays to strings of '0's and '1's
    y_true_str = ''.join(map(str, y_true.astype(int)))
    y_pred_str = ''.join(map(str, y_pred.astype(int)))
    
    return windowdiff(y_true_str, y_pred_str, window_size)