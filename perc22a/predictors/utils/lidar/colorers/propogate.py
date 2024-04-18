'''propogate.py

This file contains utility methods for propogating cone colors from an
initial starting cone to the next cone in that side of the track

All propogate functions will take in as input in a seed-cone position 
as a (1, 3) vector and a np.ndarray of size (k, 3) of the remaining cones.

All propogate functions will return an index of the next cone among the k
to propogate the color to.
'''


def propogate_naive(seed_cone_pos):
    pass