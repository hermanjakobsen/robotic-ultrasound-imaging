import numpy as np
import transforms3d.quaternions as quat

def q_log(q):
    """
    Calculates the quaternion logarithm as defined in eq. (19) in https://ieeexplore.ieee.org/document/6907291

    Args:
        q (np.array): quaternion given as (w, x, y, z) where w is the real (scalar part), and (x, y, z) are the complex (vector) part. 

    Returns:
        np.array: quaternion logarithm
    """
    v = np.clip(q[0], -1, 1)    # Quick fix to deal with somewhat unstable quaternion measurements/normalizations (i.e. v > 1 and v < -1)
    u = q[1:]
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        return np.zeros(3) 

    return np.arccos(v) * u / u_norm


def difference_quat(q1, q2):
    """
    Calculates the difference between two quaternions

    Args:
        q1 (np.array): quaternion
        q2 (np.array): quaternion  

    Returns:
        float: difference
    """
    q2_conj = quat.qconjugate(q2)
    return quat.qmult(q1, q2_conj)
    

def distance_quat(q1, q2):
    """
    Calculates distance metric between two quaternions as defined in eq. (20) in https://ieeexplore.ieee.org/document/6907291

    Args:
        q1 (np.array): quaternion given as (w, x, y, z) where w is the real (scalar part), and (x, y, z) are the complex (vector) part. 
        q2 (np.array): quaternion given as (w, x, y, z) where w is the real (scalar part), and (x, y, z) are the complex (vector) part. 

    Returns:
        float: distance metric
    """
    q_mult = difference_quat(q1, q2)

    if q_mult[0] == -1 and q_mult[1:] == np.array([0, 0, 0]):
        return 0
    
    dist = 2 * np.linalg.norm(q_log(q_mult))

    if dist > np.pi:
        dist = abs(2 * np.pi - dist)

    return dist
