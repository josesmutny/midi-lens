import pytest

from midi_lens.__helpers import Range


def test_Range():

    r1 = Range()
    # assert instance was created
    assert(isinstance(r1, Range))

    # test for ints
    r1 += -10
    r1 += 0
    r1 += 10
    assert(r1.min == -10)
    assert(r1.max == 10)
    assert(r1.range() == 20)
    assert(r1.avg() == ((-10 + 0 + 10) / 3))

    # test for floats
    r2 = Range()
    vals = [0, 0.5, 0.10, 0.123, 0.731]
    val_sum = 0
    count = len(vals)
    for val in vals:
        r2 += val
        val_sum += val

    assert(r2.min == 0)
    assert(r2.max == 0.731)
    assert(r2.avg() == val_sum / count)
    assert(r2.range() == 0.731)

    # check for beginning
    r3 = Range(min=100, max=1200)

    assert(isinstance(r3, Range))

    r3 += 200
    r3 += 150

    assert(r3.min == 100)
    assert(r3.max == 1200)
    assert(r3.avg() == 175)
    assert(r3.range() == 1100)
