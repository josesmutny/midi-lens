import pytest
from midi_lens.Visualizer import Visualizer as Visualizer
import pandas as pd
from pandas import RangeIndex
import itertools


def test___init__():
    data = pd.DataFrame((3, 5))
    desc = {'a': 'b'}
    details = {'c': 'd'}
    visualizer = Visualizer(
        data,
        desc,
        details)

    assert(isinstance(visualizer, Visualizer))
    assert(visualizer._data.equals(data))
    assert(visualizer._desc == desc)
    assert(visualizer._details == details)


raw_data = {
    'int': [0, 1, 2, 3],
    'float': [0.0, 1.5, 2.5, 4.5],
    'str': ['a', 'alfa', 'beta', 'gamma'],
    'index': [0, 1, 2, 3]
}


_data = pd.DataFrame.from_dict(raw_data)
_columns = _data.columns


@pytest.fixture
def visualizer():
    return Visualizer(
        _data,
        {'int': 'int', 'str': 'str', 'float': 'float', 'index': 'index'},
        {'int': 'int', 'str': 'str', 'float': 'float', 'index': 'index'}
    )

# Test show cols ##########################################################


_cols = []
for _comb_length in range(1, len(_columns)):
    _combinations = itertools.combinations_with_replacement(
        _columns, _comb_length)
    for combination in _combinations:
        _cols.append(combination)

_params = [*itertools.permutations(
    _columns, 3)]

# add repetitions
_params.append(('int', 'int', 'int'))
_params.append(('index', 'index', 'index'))

_concatenated_data = itertools.product(_cols, _params)
_concatenated_data = [(val[0], *val[1]) for val in _concatenated_data]


@pytest.mark.parametrize("cols,x,size,color", _concatenated_data)
def test_show_cols(visualizer, cols, x, size, color):
    if 'str' in (size, color) and not isinstance(cols, str):
        with pytest.raises(ValueError):
            visualizer.show_cols(
                "title",
                cols,
                x,
                size,
                1,
                color,
                show=False)
    else:
        visualizer.show_cols(
            "title",
            cols,
            x,
            size,
            1,
            color,
            show=False)


test_params = (
    (('str')),
    (('index')),
    (('str', 'index')),
    (('index', 'index')),
    (('str', 'int', 'float')),
    (('str', 'str', 'int', 'str')),
)

# Test scatter 3d #######################################################
_combinations = itertools.combinations_with_replacement(_columns, 5)


@pytest.mark.parametrize("x,y,z,color,hover_title", _combinations)
def test_show_scatter_3d(visualizer, x, y, z, color, hover_title):
    visualizer.show_cols_3d(
        x, y, z,
        color,
        hover_title,
        "title",
        show=False)


# test show matrix ######################################################

_cols = []
for _comb_length in range(len(_columns)):
    _combinations = itertools.combinations_with_replacement(
        _columns, _comb_length)
    for _combination in _combinations:
        _cols.append(_combination)


@pytest.mark.parametrize('cols', _cols)
def test_test(visualizer, cols):

    visualizer.show_matrix(
        cols,
        'title',
        'index',
        'index',
        show=False)

    visualizer.show_matrix(
        cols,
        'title',
        'str',
        'index',
        show=False)

    visualizer.show_matrix(
        cols,
        'title',
        'index',
        'str',
        show=False)

# test valid data ##########################################################


test___valid_data = (
    # valid
    (('str', 'float', 'int'), True),
    (('str', 'float', 'int', 'int'), True),
    (('str', 'str'), True),
    (('str'), True),
    (('index', 'index'), True),
    (('str', 'float', 'int', 'index'), True),
    (('str', 'float', 'str', 'float', 'str', 'index'), True),
    # invalid
    (('tmp'), False),
    (('tmp', 'index'), False),
    (('float', 'str', 'str', 'tmp'), False),
)


@pytest.mark.parametrize("params,expected", test___valid_data)
def test___valid(visualizer, params, expected):
    if isinstance(params, str):
        params = [params]
    assert(visualizer._Visualizer__valid(*params) == expected)
