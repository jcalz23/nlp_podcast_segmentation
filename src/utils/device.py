import torch

def get_device():

    """
    Determine the best available device for computation.

    Returns:
        torch.device: The selected device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")