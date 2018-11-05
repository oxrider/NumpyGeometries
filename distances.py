import numpy as np

def projection(p, a, b):
    # Projects point p to the line connecting a and b
    AP = p - a
    AB = b - a
    return a + np.dot(AP, AB) / np.dot(AB, AB) * AB

def point2segment(p, a, b):
    # Computes the distance between point p to the line segment connecting a and b
    proj = projection(p, a, b)
    AB = b - a
    AB_norm = np.linalg.norm(AB)
    if np.linalg.norm(proj - a) > AB_norm:
        # proj is outside AB, on b-side
        return np.linalg.norm(p-b)
    if np.linalg.norm(proj - b) > AB_norm:
        # proj is outside AB, on a-side
        return np.linalg.norm(p-a)
    # p is within AB
    return np.linalg.norm(p - proj)

