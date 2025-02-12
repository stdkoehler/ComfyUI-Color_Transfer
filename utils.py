import numpy as np


def EuclideanDistance(detected_color, target_colors):
    return np.linalg.norm(detected_color - target_colors, axis=1)


def ManhattanDistance(detected_color, target_colors):
    return np.sum(np.abs(detected_color - target_colors), axis=1)


def CosineSimilarity(detected_color, target_colors):
    return -np.dot(target_colors, detected_color) / (np.linalg.norm(detected_color) * np.linalg.norm(target_colors, axis=1))


'''def HSV_Color_Similarity(detected_colors, target_colors):
    h1, s1, _ = detected_colors
    h2, s2, _ = target_colors
    
    h1_rad, h2_rad = np.radians(h1), np.radians(h2)
    
    v1 = np.array([s1 * np.cos(h1_rad), s1 * np.sin(h1_rad)])
    v2 = np.array([s2 * np.cos(h2_rad), s2 * np.sin(h2_rad)])
    
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))'''