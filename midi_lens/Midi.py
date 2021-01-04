import pretty_midi
import pandas as pd
import numpy as np
from collections import defaultdict
import pychord
from .__helpers import Range
from os.path import basename as basename
# import warnings

# warnings.filterwarnings('ignore')


class Midi:
    """
    Import and process MIDI files

    ...

    Attributes
    ----------
    trans : dict
        dict with labels for analysis output
    full_trans: dict
        Same as trans, but for get_data()['timelapse']

    Methods
    -------
    get_stats(full=False)
        Retrieve statistcs for file loaded on __init__
    """

    full_trans = (
        {
            'codes': 'MIDI codes',
            'nice_codes': 'Note values',
            'vel': 'Velocity',
            'len': 'Length [s]',
            'len_diff': 'Change in length [s]',
            'w_vel': 'Weighted velocity',
            'tone_range': 'Tone range [semitone]',
            'poly': 'Polyphony',
            'code_range': 'MIDI code range',
            'vel_range': 'Velocity range',
            'time': 'Time [s]'
        }, {
            'codes': 'MIDI codes (21 = A0)',
            'nice_codes': 'Note values (e.g.: A#4)',
            'vel': 'Note velocity (force of notes)',
            'len': 'Note duration in seconds',
            'len_diff': 'Change in note duration',
            'w_vel': 'Weighted velocity (vel * log(len))',
            'tone_range': 'Tone range [semitone]',
            'poly': 'Number of notes pressed at once',
            'code_range': 'Difference between lowest and highest tone',
            'vel_range': 'Difference between quitest and loudest note',
            'time': 'Time since beginning of piece'
        }
    )

    _desc = {
        'weighted_vel': 'Weighted velocity',
        'tempo_diff_avg': 'Mean tempo change',
        'tempo_max_diff': 'Max tempo change',
        'tempo_min_diff': 'Min tempo change',
        'tempo_range': 'Range of tempi',
        'avg_tempo': 'Tempo',
        'tempo_len_range': 'Tempo length range',
        'avg_tempo_len': 'Tempo length',
        'avg_vel_range': 'Velocity range',
        'avg_avg_vel': 'Velocity',
        'std_vel_range': 'Range of std velocity',
        'std_avg_vel': 'Std of mean velocity',
        'avg_tone_range': 'Tone range',
        'avg_avg_tone': 'Mean tones',
        'std_tone_range': 'Std tone range',
        'std_avg_tone': 'Std tone',
        'len_diff_range': 'Range of length changes',
        'avg_len_diff': 'Length changes',
        'avg_poly': 'Mean polyphony',
        'poly_range': 'Polyphony range',
        'total_len': 'Piece length',
        'name': 'Name'
    }

    _details = {}
    _details.update(_desc)
    _details['name'] = 'Name of piece'
    _details['avg_poly'] = 'Mean number of notes played at once'
    _details['weighted_vel'] = 'Weighted velocity (vel * log(len))'

    for key, val in _details.items():
        if 'temp' in key:
            _details[key] = val + ' [bpm]'
        elif 'tone' in key:
            _details[key] = val + ' [semitones]'
        elif 'len' in key:
            _details[key] = val + ' [s]'

    trans = (_desc, _details)

    def __init__(self,
                 file_name,
                 fs: float,
                 no_pedal=False,
                 ignore_breaks=False):
        """Construct Midi object

        Args:
            file_name (file): path to valid .mid file
            fs (float): frequency of time samples (width of sample = 1s/fs) 
            no_pedal (bool, optional): If true, ignore pedal signals. Defaults to False.
            ignore_breaks (bool, optional): If true, ignore silence. Defaults to False.
        """
        self._data = pretty_midi.PrettyMIDI(file_name,)
        self._data.remove_invalid_notes()
        self._fs = fs
        self._name = basename(file_name)
        self._ignore_breaks = ignore_breaks
        self._pedal_threshold = 64  # default threshold
        if no_pedal:
            self._pedal_threshold = None  # ignore pedal signals

    def __analyze_tempi_diff(self) -> dict:
        """Store statistics about changes in tempo into dictionary.\n
        Keys:
            tempo_diff_avg  mean size of changes in tempo   [bpm]
            tempo_max_diff  largest change of tempo         [bpm]
            tempo_min_diff  smallest change of tempo        [bpm]

        Returns:
            dict: Dictionary of statistical values
        """
        last_tempo = 0
        max_diff = 0
        min_diff = float('inf')
        diff_sum = 0
        tempi = self._data.get_tempo_changes()[1]
        for tempo in tempi:
            curr_diff = abs(last_tempo - tempo)

            # min and max comparisons
            min_diff = min(min_diff, curr_diff)
            max_diff = max(max_diff, curr_diff)

            # add to total
            diff_sum += curr_diff
            last_tempo = tempo

        # in case there is only one tempo
        # throughout the piece
        tempo_diff_avg = 0
        if len(tempi) != 1:
            tempo_diff_avg = diff_sum / len(tempi) - 1

        # if there where no differences, then
        # set the smallest diference to 0
        if min_diff == float('inf'):
            min_diff = 0
            max_diff = 0

        return {
            'tempo_diff_avg': tempo_diff_avg,
            'tempo_max_diff': max_diff,
            'tempo_min_diff': min_diff
        }

    def __analyze_tempi_base(self, full: bool) -> dict:
        """Store statistics about values of tempo into dictionary\n
        Keys:
            min_tempo      [bpm] slowest tempo
            max_tempo      [bpm] fastest tempo
            avg_tempo      [bpm] weighted average of tempo
            min_tempo_len  [s]   shortest tempo duration throughout piece
            max_tempo_len  [s]   longest tempo throughout piece without interruption
            avg_tempo_len  [s]   average tempo duration
        Args:
            full (bool): if true, gather detailed piece-wide metrics
        Returns:
            dict: Dictionary with statistical values
        """
        tempo = Range()
        tempo_len = Range()

        tempo_changes = self._data.get_tempo_changes()
        tempi = {}
        old_time = 0
        for curr_time, curr_tempo in zip(*tempo_changes):

            # min and max comparisons for tempo
            tempo += curr_tempo

            tempo_length = curr_time - old_time
            curr_tempo = int(round(curr_tempo))

            # min max comparisons for tempo length
            tempo_len += tempo_length

            # add curr tempo length eighter to existing key,
            # or add the key, if it did not exist before
            if curr_tempo in tempi:
                tempo_length += tempi[curr_tempo]
            tempi[curr_tempo] = tempo_length
            old_time = curr_time

        # add the length of the last tempo
        tempi[round(tempo_changes[1][-1])
              ] += self._data.get_end_time() - old_time

        # create dataframe from dictionary
        tempi_df = pd.DataFrame.from_dict(
            tempi, orient="index", columns=["Duration"])

        # calculate tempo avg
        weighted_tempo_sum = sum(tempi_df.index * tempi_df.Duration)
        avg_tempo = weighted_tempo_sum / sum(tempi_df.Duration)

        # calculate avg tempo duration
        avg_tempo_len = tempi_df.Duration.mean()

        # remove tempos lasting less than 5% of whole
        min_duration = (5 * 100) / self._data.get_end_time()
        tempi_df = tempi_df.drop(
            tempi_df[tempi_df.Duration < min_duration].index)
        tempi_df['Percentage'] = tempi_df.Duration * \
            100 / self._data.get_end_time()

        # sort resulting tempo distribution
        tempi_df.sort_index(ascending=True, inplace=True)

        stats = {
            'tempo_range': tempo.range(),
            'avg_tempo': avg_tempo,
            'tempo_len_range': tempo_len.range(),
            'avg_tempo_len': avg_tempo_len
        }

        if full:
            stats['tempo_stats'] = tempi_df

        return stats

    def __analyze_tempi(self, full=False) -> dict:
        """Produces a Series containing tempo statistics\n
       Returns:
            dict: Dictionary of tempo statistics
        """
        stats = {}
        stats.update(self.__analyze_tempi_diff())
        stats.update(self.__analyze_tempi_base(full))

        return stats

    def __analyze_single_notes(self, notes: np.array) -> pd.DataFrame:
        """Generate velocity statistics from single note data
        Input notes should begin from midi code 21 (represented as note 0)

        Args:
            notes (np.array): data of notes

        Returns:
            dict: Pandas dataframe with note tone, and velocity statistics
        """

        stats = defaultdict(lambda: defaultdict(int))

        for midi_code, note in enumerate(notes, start=21):
            non_zero = np.array(note[note > 0])
            if len(non_zero):
                stats[midi_code]['min_vel'] = non_zero.min()
                stats[midi_code]['max_vel'] = non_zero.max()
                stats[midi_code]['avg_vel'] = round(non_zero.mean())

        return pd.DataFrame(stats).T

    def __analyze_notes_base(
            self,
            notes: np.array,
            midi_codes: list,
            full: bool):
        """Gather statistics about harmony changes through time
        Keys:
            avg_vel_range   average of velocity range for each frame
            avg_avg_vel     average of average velocity for each frame
            std_vel_range   std deviation of velocity ranges
            std_avg_vel     std deviation of average velocity ranges
            avg_tone_range  average tonal range for each frame
            avg_avg_tone    average of average tone of each frame
            std_tone_range  std deviation of tonal ranges
            std_avg_tone    std deviation of average tone
            len_diff_range  difference between minimal and maximal note
                            change (speed contrast)
            avg_len_diff    average change in speed
            avg_poly        average number of notes pressed on one instant
            poly_range      difference between min number of pressed keys,
                            and max number of pressed keys
        Args:
            notes (2-dimensional array): array where each row is 1/fs of a second
            midi_codes (array): array of used midi codes (from 21 to 109)
            fs (int, optional): Granularity of time division. Defaults to 100.
            full (bool): if true, include detailed note and range changes
        Returns:
            defaultdict(int): Dictionary containing statistical values
        """

        # add empty row at final position,
        # so that all notes are accounted for in for loop
        notes = np.hstack((notes, np.zeros((88, 1))))
        notes = notes.T  # transpose, so that each row represents a 1/fs of a second
        notes = notes[:, midi_codes-21]  # trim for used notes

        # First compress data from pianororll, so that each
        # row is different from the next one, and so that
        # index indicates the time from the beginning
        slices = pd.DataFrame(notes, index=range(notes.shape[0]))
        # remove all but last contiguous duplicate
        raw_data = slices.loc[(slices.shift() != slices).any(axis=1)]
        # rename columns to be midi_codes (instead of indexes of midi_codes)
        raw_data.columns = midi_codes[raw_data.columns]

        # add column with midi codes pressed for given row
        codes = []
        vel_vals = []
        for _, row in raw_data.iterrows():
            curr_codes = tuple(midi_codes[row.astype('bool')])
            curr_vel_vals = tuple(row[row.astype('bool')])

            codes.append(curr_codes)
            vel_vals.append(curr_vel_vals)

        # add before adding next column
        data = pd.DataFrame(index=raw_data.index)
        processed = pd.DataFrame(index=data.index)

        data['vel_vals'] = pd.Series(vel_vals, index=data.index)
        data['codes'] = pd.Series(codes, index=data.index)

        # compute stats based on data ##############################################

        def tuple_range(x): return 0 if not len(x) else max(x) - min(x)

        processed['codes'] = data.codes.apply(
            lambda x: 0 if not len(x) else np.mean(x))
        midi_nice_codes = ['A', 'A#', 'B', 'C', 'C#', 'D',
                           'D#', 'E', 'F', 'F#', 'G', 'G#']
        processed['nice_codes'] = processed.codes.fillna(0).apply(
            lambda x: str(midi_nice_codes[(round(x) - 21) % 12]) + str((round(x) - 21)//12) if round(x) > 0 else 'Silence')
        processed['vel'] = data.vel_vals.apply(
            lambda x: 0 if not len(x) else np.mean(x))
        # Lenght is calculated as the differences of indexes.
        length = np.array(pd.Series(data.index).diff())
        length = length[1:]
        # plus add the lenght of the last note as the difference
        # between the last index and the total lenght of the piece
        length = np.append(length, notes.shape[0] - data.index.max())
        processed['len'] = length

        # add column for changes in length.
        processed['len_diff'] = np.array(processed.len.diff())

        # Velocity data weighted by log of note length
        processed['w_vel'] = processed.vel * np.log(processed.len)

        # difference in tones for each slice
        processed['tone_range'] = data.codes.apply(tuple_range)

        # number of notes in each slice. Breaks have poly=0
        processed['poly'] = data.codes.apply(lambda x: len(x))

        # Fill all nan values with 0.
        processed = processed.fillna(0)

        # if ignore_breaks is set, ignore slices where no note was being played
        if self._ignore_breaks:
            processed = processed[processed.poly > 0]

        # add range columns for tone and vel
        processed['code_range'] = data.codes.apply(
            lambda x: 0 if not len(x) else np.max(x) - np.min(x))
        processed['vel_range'] = data.vel_vals.apply(tuple_range)

        # modify time-aware columns so that their time is in seconds
        processed.index /= self._fs
        processed.len /= self._fs

        # Done gathering stats #######################################################

        note_stats = {
            'weighted_vel': processed.w_vel.mean(),
            'avg_vel_range': processed.vel.max() - processed.vel.min(),
            'avg_avg_vel': processed.vel.mean(),
            'std_vel_range': processed.vel_range.std(),
            'std_avg_vel': processed.vel.std(),
            'avg_tone_range': processed.code_range.mean(),
            'avg_avg_tone': processed.codes.mean(),
            'std_tone_range': processed.code_range.std(),
            'std_avg_tone': processed.codes.std(),
            'len_diff_range': processed.len_diff.max() - processed.len_diff.min(),
            'avg_len_diff': processed.len_diff.mean(),
            'avg_poly': processed.poly.mean(),
            'poly_range': processed.poly.max() - processed.poly.min(),
            'total_len': notes.shape[0] / self._fs
        }
        if full:
            note_stats['timelapse'] = processed
        return note_stats

    def __analyze_notes(self, full: bool) -> dict:
        """Produces a Series containing note statistics
        Keys:
            avg_vel_range   average of velocity range for each frame
            avg_avg_vel     average of average velocity for each frame
            std_vel_range   std deviation of velocity ranges
            std_avg_vel     std deviation of average velocity ranges
            avg_tone_range  average tonal range for each frame
            avg_avg_tone    average of average tone of each frame
            std_tone_range  std deviation of tonal ranges
            std_avg_tone    std deviation of average tone
            len_diff_range  difference between minimal and maximal note
                            change (speed contrast)
            avg_len_diff    average change in speed

        Args:
            full (bool): if true, gather detailed piece-wide metrics

        Returns:
            Dictionary containing note statistics
        """

        # rows represent a histogram of the single note through time
        # with a frequency of 0.01 seconds
        # leave only rows 21 to 108, which are the valid notes for a standard 88 key piano
        notes = self._data.get_piano_roll(
            fs=self._fs,
            pedal_threshold=self._pedal_threshold)[21:109, :]

        """ Piano roll format
            row      content
            21 (A0): 0  0  0  0  0  0  0
            ...
            88 (C8): 0 21 21 21 12  0  0
            where each column represents 0.01 seconds of real play,
            and the values of rows indicate the velocity of the key
            (thus C8 was played 2: once at 21 and once at 12)
        """

        stats = {}

        midi_codes = np.array(self.__analyze_single_notes(notes).index)

        stats.update(self.__analyze_notes_base(notes, midi_codes, full))

        return stats

    def get_stats(self, full=False) -> pd.DataFrame:
        """Produce tempo and note stats.
            Keys            Unit  Description
            tempo_diff_avg  [bpm] mean size of changes in tempo
            tempo_max_diff  [bpm] largest change of tempo
            tempo_min_diff  [bpm] smallest change of tempo
            min_tempo_len   [s]   shortest tempo duration throughout piece
            max_tempo_len   [s]   longest tempo throughout piece without interruption
            avg_tempo_len   [s]   average tempo duration
            min_tempo       [bpm] slowest tempo
            max_tempo       [bpm] fastest tempo
            avg_tempo       [bpm] weighted average of tempo
            avg_vel_range         average of velocity range for each frame
            avg_avg_vel           average of average velocity for each frame
            std_vel_range         std deviation of velocity ranges
            std_avg_vel           std deviation of average velocity ranges
            avg_tone_range        average tonal range for each frame
            avg_avg_tone          average of average tone of each frame
            std_tone_range        std deviation of tonal ranges
            std_avg_tone          std deviation of average tone
            len_diff_range        difference between minimal and maximal note
                                  change (speed contrast)
            avg_len_diff          average change in speed
            poly_range            difference between min number of pressed keys,
                                  and max number of pressed keys
            total_len       [s]   Total length of piece

        Args:
            full (bool): if true, gather detailed piece-wide metrics

        Returns:
            dict: Dictionary of statistical data
        """
        stats = self.__analyze_tempi(full)
        stats.update(self.__analyze_notes(full))
        return stats
