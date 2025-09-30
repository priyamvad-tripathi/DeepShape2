from functools import reduce

__all__ = [
    "time_string",
]


def time_string(t):
    return "%02d:%02d:%02d.%03d" % reduce(
        lambda ll, b: divmod(ll[0], b) + ll[1:], [(round(t * 1000),), 1000, 60, 60]
    )
