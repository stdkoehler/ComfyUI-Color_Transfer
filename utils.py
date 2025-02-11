import numpy as np


def EuclideanDistance(detected_colors, target_colors):
    return np.linalg.norm(detected_colors - target_colors, axis=1)


def ManhattanDistance(detected_colors, target_colors):
    return np.sum(np.abs(detected_colors - target_colors), axis=1)
