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

    v = q[0] if q[0] < 1 else 2 - q[0]     # Quick fix to deal with somewhat unstable quaternion measurements
    u = q[1:]
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        return np.zeros(3) 

    return np.arccos(v) * u / u_norm
    


def q_dist(q1, q2):
    """
    Calculates distance metric between two quaternions as defined in eq. (20) in https://ieeexplore.ieee.org/document/6907291

    Args:
        q1 (np.array): quaternion given as (w, x, y, z) where w is the real (scalar part), and (x, y, z) are the complex (vector) part. 
        q2 (np.array): quaternion given as (w, x, y, z) where w is the real (scalar part), and (x, y, z) are the complex (vector) part. 

    Returns:
        float: distance metric
    """
    q2_conj = quat.qconjugate(q2)
    q_mult = quat.qmult(q1, q2_conj)

    if q_mult[0] == -1 and q_mult[1:] == np.array([0, 0, 0]):
        return 0
    
    dist = 2 * np.linalg.norm(q_log(q_mult))

    if dist > np.pi:
        dist = abs(2 * np.pi - dist)

    return dist
