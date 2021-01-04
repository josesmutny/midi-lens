import pytest
import pandas as pd
import itertools
from os.path import join, basename, splitext
from glob import glob

from midi_lens.Midi import Midi
from midi_lens.midi_lens import analysis_from_json, show_data, show_full_data, main

_path_to_analysis = 'midi_lens/tests/data/analysis.json'

_params = (
    (_path_to_analysis, False),
    ('pizza.json', True),
    ('pizza', True))


@pytest.mark.parametrize("path,empty", _params)
def test_analysis_from_json(path, empty):
    result = analysis_from_json(path)

    assert(isinstance(result, pd.DataFrame))
    if empty:
        assert result.empty
    else:
        assert result.shape[0] > 0 and result.shape[1] > 0


_path_to_data = 'midi_lens/tests/data'
_paths = (
    'midi_lens/tests/data/chopin_etude_10_3.mid',
    'midi_lens/tests/data/mz_331_3.mid',
    'midi_lens/tests/data/chpn_op25_e12.mid'
)
_args = itertools.permutations((False, True), 2)

_params = [*itertools.product(_paths, _args)]

print(_params)


@pytest.mark.parametrize("path,args", _params)
def test_show_full_data(path, args):
    file = Midi(path, 23)

    stats = file.get_stats(full=True)['timelapse']

    stats['time'] = stats.index

    show_full_data(
        file.get_stats(full=True)['timelapse'],
        'name',
        args[0],
        args[1],
        False,
        False
    )


_args = itertools.permutations((False, True), 2)
_args = [*_args]


@pytest.mark.parametrize("args", _args)
def test_show_data(args):

    files = glob(join(_path_to_data, '*.mid'))

    data_raw = []
    file_names = []

    for filename in files:
        file = Midi(filename, 23)
        stats = file.get_stats()
        nice_name = splitext(basename(filename))[0]
        data_raw.append(stats)
        file_names.append(nice_name)

    data = pd.DataFrame(data_raw, index=file_names)

    data['name'] = data.index

    show_data(
        stats,
        'name',
        args[0],
        args[1],
        False,
        False)
