"""Freeze Python structural containers while borrowing numeric array leaves."""


def tuples(value):
    if isinstance(value, (list, tuple)):
        return tuple(tuples(item) for item in value)
    return value
