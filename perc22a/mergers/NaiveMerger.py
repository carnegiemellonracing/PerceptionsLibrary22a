'''Naive Merger

sufficient() -> true iff non-zero number of cones have been added
merge() -> returns all cones added since last reset() (doesn't remove dups)
'''

from perc22a.mergers.MergerInterface import Merger


class NaiveMerger(Merger):

    def __init__(self):
        pass

    pass