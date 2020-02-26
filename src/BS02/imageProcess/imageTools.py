import numpy as np

def eDistance(p1, p2):
    """
    function description:
    get Euclidean distance  between point p1 and p2

    :param p1: point
    :param p2: point
    :return: distance
    """
    distance = np.sqrt(np.sum((p1 - p2) ** 2))
    return distance


