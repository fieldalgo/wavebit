from .emsolver.rcwa import RCWANumpy


class SolverNumpy(RCWANumpy):

    def __init__(self, *args, **kwargs):
        RCWANumpy.__init__(self, *args, **kwargs)

