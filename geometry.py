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

def triangle_areas(tris):
    # Computes areas of triangles
    # Args: tris N x 3 x 3
    tris = tris.reshape((-1,3,3))
    sides = np.linalg.norm(tris - tris[:,[1,2,0],:], axis=-1)
    s = sides.sum(axis=-1)/2
    return np.sqrt(s*(s-sides[:,0])*(s-sides[:,1])*(s-sides[:,2]))

def angle_between_vectors(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cos = np.dot(v1, v2)
    sin = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sin, cos)