import pytest
import glob
import os
import pandas as pd
import numpy as np
from pretty_midi import PrettyMIDI

from midi_lens.Midi import Midi

files = glob.glob(os.path.join('midi_lens/tests/data/', '*.mid'))


@pytest.mark.parametrize("path", [file for file in files])
def test_transitions(path):
    m1 = Midi(path, 23)

    assert(isinstance(m1, Midi))

    # check details and validity of translation info
    assert(isinstance(Midi.trans, tuple))
    assert(isinstance(Midi.full_trans, tuple))

    assert(isinstance(Midi.trans[0], dict))
    assert(isinstance(Midi.trans[1], dict))
    assert(isinstance(Midi.full_trans[0], dict))
    assert(isinstance(Midi.full_trans[1], dict))

    assert(set(Midi.trans[0].keys()) == set(Midi.trans[1].keys()))
    assert(set(Midi.full_trans[0].keys()) == set(Midi.full_trans[1].keys()))


@pytest.mark.parametrize("args,expected", [
    ([1, True, True], [1, None, True]),
    ([100, False, False], [100, 64, False]),
    ([12, False, True], [12, 64, True]),
    ([13, True, False], [13, None, False])])
def test_instantiation(args, expected):
    for path in files:
        m1 = Midi(
            path,
            args[0],
            args[1],
            args[2])
        assert(isinstance(m1._data, PrettyMIDI))
        assert(m1._name == os.path.basename(path))
        assert(m1._pedal_threshold == expected[1])
        assert(m1._ignore_breaks == expected[2])


@pytest.mark.parametrize("path", files)
def test___analyze_tempi_diff(path):
    m1 = Midi(path, 23)

    result = m1._Midi__analyze_tempi_diff()

    assert(isinstance(result, dict))
    assert(set(result.keys()) == set(
        ['tempo_diff_avg', 'tempo_max_diff', 'tempo_min_diff']))

    # check value validity
    assert(result['tempo_max_diff'] != 0)
    assert(result['tempo_min_diff'] != float('inf'))


@pytest.mark.parametrize("path", files)
def test___analyze_tempi_base(path):
    for full in (True, False):
        m1 = Midi(path, 23)

        result = m1._Midi__analyze_tempi_base(full)

        assert(isinstance(result, dict))
        if full:
            assert(set(result.keys()) == set(
                ['tempo_range',
                 'avg_tempo',
                 'tempo_len_range',
                 'avg_tempo_len',
                 'tempo_stats']))
            assert(isinstance(result['tempo_stats'], pd.DataFrame))

            assert(set(result['tempo_stats'].columns)
                   == set(['Duration', 'Percentage']))
            assert(result['tempo_stats'].shape[0] > 0)

        else:
            assert(set(result.keys()) == set(
                ['tempo_range',
                 'avg_tempo',
                 'tempo_len_range',
                 'avg_tempo_len']))

    # check validity of values
    assert(result['tempo_range'] >= 0)
    assert(result['avg_tempo'] > 0)
    assert(result['tempo_len_range'] >= 0)
    assert(result['avg_tempo_len'] > 0)


@pytest.mark.parametrize("path", files)
def test___analyze_tempi(path):
    m1 = Midi(path, 23)

    result = m1._Midi__analyze_tempi()

    assert(isinstance(result, dict))
    assert(set(result.keys()) == set([
        'tempo_diff_avg',
        'tempo_max_diff',
        'tempo_min_diff',
        'avg_tempo_len',
        'tempo_len_range',
        'tempo_range',
        'avg_tempo',
    ]))

    for val in result.values():
        assert(val >= 0)


@pytest.mark.parametrize("path", files)
def test___analyze_single_notes(path):
    m1 = Midi(path, 23)

    data = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    non_zero = np.array([[x for x in range(1, 11)]])
    result = m1._Midi__analyze_single_notes(data)

    assert(isinstance(result, pd.DataFrame))

    assert(set(result.index) == set([21]))
    assert(result.loc[21, 'min_vel'] == 1)
    assert(result.loc[21, 'max_vel'] == 10)
    assert(result.loc[21, 'avg_vel'] == round(non_zero.mean()))


def test___analyze_notes_base():
    m1 = Midi(files[0], 1)

    notes = np.array([[0, 1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0, 4, 4, 4], ]*88)
    midi_codes = np.array([30, 40, 50])
    full = True

    result = m1._Midi__analyze_notes_base(
        notes,
        midi_codes,
        full
    )

    assert(isinstance(result, dict))
    for key, val in result.items():
        if key in ('avg_len_diff', 'timelapse'):
            continue
        print(key)
        assert(val >= 0)

    base_notes = np.array(
        [0, 1, 0, 2, 0, 3, 0, 4, 0])

    avg_tones = np.array([0, 40, 0, 40, 0, 40, 0, 40, 0])

    assert(result['avg_vel_range'] == 4)
    assert(result['avg_avg_vel'] == base_notes.mean())
    # set delta degrees of freedom to 1 to calculate std as
    # pandas does
    assert(result['std_vel_range'] == 0)
    assert(result['std_avg_vel'] == base_notes.std(ddof=1))
    assert(result['avg_tone_range'] == ((0 * 5) + (20 * 4))/(5 + 4))
    assert(result['avg_avg_tone'] == avg_tones.mean())
    assert(result['std_tone_range'] == np.array(
        [0, 0, 0, 0, 0, 20, 20, 20, 20]).std(ddof=1))
    assert(result['std_avg_tone'] == avg_tones.std(ddof=1))
    # changing from len = 3 to len = 1
    assert(result['len_diff_range'] == abs(-2 - 2))
    assert(result['avg_len_diff'] == np.array([-2, 2]*4).mean())
    assert(result['avg_poly'] == ((0 * 5) + (3 * 4)) / 9)
    assert(result['poly_range'] == 3)
    assert(result['total_len'] == len(notes[0]) + 1)


@pytest.mark.parametrize("path", files)
def test_get_stats(path):
    m1 = Midi(path, 23)

    result1 = m1.get_stats(full=False)

    assert(isinstance(result1, dict))
    assert('timelapse' not in result1.keys())

    result2 = m1.get_stats(full=True)
    assert(isinstance(result2, dict))
    assert('timelapse' in result2.keys())
    assert(isinstance(result2['timelapse'], pd.DataFrame))
