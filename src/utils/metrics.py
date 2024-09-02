import numpy as np
from nltk.metrics.segmentation import windowdiff

def windiff(y_true, y_pred, window_size, attention_mask=None):
    """
    Calculate the WindowDiff metric for segmentation evaluation using NLTK.

    Args:
        y_true (np.array or list): Ground truth segmentation.
        y_pred (np.array or list): Predicted segmentation.
        window_size (int): Size of the comparison window.
        attention_mask (np.array or list, optional): Attention mask to apply. Defaults to None.

    Returns:
        float: WindowDiff score (lower is better).
    """
    # Convert inputs to numpy arrays if they're not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if attention_mask is not None:
        attention_mask = np.array(attention_mask)

    if attention_mask is not None:
        y_true = y_true[attention_mask.astype(bool)]
        y_pred = y_pred[attention_mask.astype(bool)]

    # Convert numpy arrays to strings of '0's and '1's
    y_true_str = ''.join(map(str, y_true.astype(int)))
    y_pred_str = ''.join(map(str, y_pred.astype(int)))
    
    return windowdiff(y_true_str, y_pred_str, window_size)