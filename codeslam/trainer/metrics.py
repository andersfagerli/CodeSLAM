import numpy as np

def RMSE(input: np.ndarray, target: np.ndarray):
    return np.sqrt(np.mean((input-target)**2))

def logRMSE(input: np.ndarray, target: np.ndarray, eps=1e-6):
    input_clipped = np.clip(np.copy(input), a_min=eps, a_max=None)
    target_clipped = np.clip(np.copy(target), a_min=eps, a_max=None)

    return np.sqrt(np.mean(np.log(input_clipped)-np.log(target_clipped))**2)

def ARD(input: np.ndarray, target: np.ndarray, eps=1e-6):
    input_clipped = np.clip(np.copy(input), a_min=eps, a_max=None)
    target_clipped = np.clip(np.copy(target), a_min=eps, a_max=None)

    return np.mean(np.abs(input_clipped - target_clipped) / target_clipped)

def SRD(input: np.ndarray, target: np.ndarray, eps=1e-6):
    input_clipped = np.clip(np.copy(input), a_min=eps, a_max=None)
    target_clipped = np.clip(np.copy(target), a_min=eps, a_max=None)

    return np.mean((input_clipped - target_clipped)**2 / target_clipped)

def accuracy(input: np.ndarray, target: np.ndarray, threshold=1.25, eps=1e-6):
    input_clipped = np.clip(np.copy(input), a_min=eps, a_max=None)
    target_clipped = np.clip(np.copy(target), a_min=eps, a_max=None)

    delta = np.maximum(input_clipped / target_clipped, target_clipped / input_clipped)

    return np.mean(delta < threshold)