"""Provides entry point main()"""
from midi_lens.Visualizer import Visualizer
from midi_lens.Midi import Midi

import inspect
import warnings
import pandas as pd
import numpy as np

from tqdm import tqdm
from os.path import join, basename, exists, splitext
from glob import glob


def analysis_from_json(file) -> pd.DataFrame:
    """Scan datafram from .json file,
       and check if it contains valid statistics

    Args:
        file (os.path): name of .json file

    Returns:
        pd.DataFrame: DataFrame containing midi file stats
    """

    try:
        data = pd.read_json(file, orient='table')
    except Exception as err:
        print("Error reading from json ({})".format(err))
        return pd.DataFrame()

    # to-do
    # check if shape is ok,
    # and that columns match
    cols = ['tempo_diff_avg', 'tempo_max_diff', 'tempo_min_diff', 'tempo_range',
            'avg_tempo', 'tempo_len_range', 'avg_tempo_len', 'weighted_vel',
            'avg_vel_range', 'avg_avg_vel', 'std_vel_range', 'std_avg_vel',
            'avg_tone_range', 'avg_avg_tone', 'std_tone_range', 'std_avg_tone',
            'len_diff_range', 'avg_len_diff', 'avg_poly', 'poly_range',
            'total_len']
    if len(data.columns) != len(cols) or (data.columns != cols).all() or data.shape[0] < 1:
        return pd.DataFrame()

    print("{} successfully read".format(file))
    return data


def show_full_data(
        data: pd.DataFrame,
        name: str,
        show_detailed: bool,
        show_space: bool,
        show_custom: bool,
        show=True) -> None:
    """Display visualizations for single file.
    Format must be that of Midi.full_trans

    Args:
        data (pd.DataFrame): data to be visualized
    """
    desc, details = Midi.full_trans
    data.sort_values('w_vel')
    Viz = Visualizer(data, desc, details)
    Viz.show_cols(
        title='Linear analysis of {}, with sizes by note length.'.format(
            name),
        cols=('codes', 'vel'),
        x='time',
        color='poly',
        size='len',
        size_scale=0.006,
        show=show
    )
    if show_detailed:
        Viz.show_matrix(
            ('len', 'vel', 'w_vel', 'time', 'poly'),
            'Correlation analysis of {}'.format(name),
            'nice_codes',
            'poly',
            show=show
        )

        Viz.show_cols(
            title='Detailed linear analysis of {}, with sizes by note tone.'.format(
                name),
            cols=('codes', 'len_diff', 'tone_range', 'w_vel'),
            x='time',
            color='poly',
            size='codes',
            size_scale=1.5,
            show=show
        )

    if show_space:
        Viz.show_cols_3d(
            x='time',
            y='codes',
            z='vel',
            color='poly',
            title='Spatial analysis of {} by pitch and velocity through time.'.format(
                name),
            hover_title='nice_codes',
            show=show
        )
    if show_custom:
        Viz.custom_show(name, show)


def show_data(
        data: pd.DataFrame,
        name: str,
        show_detailed: bool,
        show_space: bool,
        show_custom: bool,
        show=True):
    """Display visualizations for batch of files

    Args:
        data (pd.DataFrame): data to be visualized
        name (str): name of analyzed files
        show_detailed (bool): true to show detailed vis.
        show_space (bool): true to show 3D vis.
        show_custom (bool): true to show custom vis.
    """

    data_desc, data_details = Midi.trans
    vis = Visualizer(data, data_desc, data_details)

    vis.show_cols(
        title='Linear analysis of files from {}, with sizes by piece length.'.format(
            name),
        cols=('avg_tempo', 'avg_avg_tone', 'total_len',
              'avg_poly'),
        x='name',
        color='weighted_vel',
        size='total_len',
        size_scale=0.75,
        show=show
    )

    if show_detailed:
        vis.show_matrix(
            ('total_len',
             'avg_avg_tone',
             'avg_avg_vel',
             'avg_len_diff',
             'avg_poly'),
            'Correlation analysis of pieces from {}'.format(name),
            'name',
            'avg_avg_vel',
            show=show
        )
    if show_space:
        vis.show_cols_3d(
            x='total_len',
            y='avg_tempo',
            z='avg_poly',
            color='avg_avg_vel',
            hover_title='name',
            title="Spatial analysis of pieces from {}".format(name),
            show=show
        )
    if show_custom:
        vis.custom_show(
            "Custom plot for pieces from {}".format(name),
            show=show)


def show_comparison(
        data: pd.DataFrame,
        name: str,
        comp_data: pd.DataFrame,
        comp_name: str,
        detailed: bool,
        space: bool,
        custom: bool,
        show=True):

    data_desc, data_details = Midi.full_trans
    vis = Visualizer(data, data_desc, data_details)

    vis.show_comparison(
        x='time',
        y='vel',
        color='poly',
        size='len',
        size_scale=0.005,
        comp_data=comp_data,
        title='Comparison of {} and {}'.format(name, comp_name),
        show=show)


def main(args) -> None:
    """Generate data for analysis, and produce visualizations
    """
    try:
        args.frequency = float(args.frequency)
    except:
        print('Invalid frequency, defaulting to 48 slices per second')
        args.frequency = 48

    if args.path.endswith('.mid'):
        if not exists(args.path):
            print("File not found.")
            exit(1)

        name = basename(args.path)

        print("Gathering data from {}".format(name))

        try:
            file = Midi(args.path,
                        fs=args.frequency,
                        no_pedal=args.no_pedal,
                        ignore_breaks=args.ignore_breaks)
            data = file.get_stats(full=True)
        except Exception as err:
            print("File could not be read:", err)
            exit(1)

        timelapse = data['timelapse']
        # add to remove need for normalization of index
        timelapse['time'] = timelapse.index
        nice_name = splitext(basename(args.path))[0]

        if args.compare and args.compare.endswith('.mid') and exists(args.compare):
            comp_name = basename(args.compare)
            comp_nice_name = splitext(comp_name)[0]
            print("Gathering data from {}".format(comp_name))

            try:
                comp_file = Midi(args.compare,
                                 fs=args.frequency,
                                 no_pedal=args.no_pedal,
                                 ignore_breaks=args.ignore_breaks)
                comp_data = comp_file.get_stats(full=True)
            except Exception as err:
                print("File could not be read:", err)
                exit(1)

            comp_timelapse = comp_data['timelapse']
            comp_timelapse['time'] = comp_timelapse.index

            show_comparison(
                timelapse,
                nice_name,
                comp_timelapse,
                comp_nice_name,
                args.detailed,
                args.space,
                args.custom
            )
            exit(0)

        show_full_data(
            timelapse,
            nice_name,
            args.detailed,
            args.space,
            args.custom)
        exit(0)

    data = pd.DataFrame()
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'

    # check if path leads to analysis file
    if args.path.endswith('.json'):
        if exists(args.path):
            print("Attempting to load analysis data from {}".format(args.path))
            data = analysis_from_json(args.path)
            if data.empty:
                exit(1)

    # if no analysis, then treat as dir and find .mid files
    if data.empty:
        files = glob(join(args.path, '*.mid'))
        print("Scanning for .mid files in {}".format(args.path))
        if not files:
            print("No .mid files were found")
            exit(1)

        print("Analyzing {} file{} from {}".format(
            len(files), 's' if len(files) != 1 else '', args.path))

        data_raw = []
        file_names = []
        log = []

        for filename in tqdm(files, unit=' files', bar_format=bar_format):
            with warnings.catch_warnings():
                warnings.filterwarnings('error')

                try:
                    file = Midi(
                        filename,
                        fs=args.frequency,
                        no_pedal=args.no_pedal,
                        ignore_breaks=args.ignore_breaks)
                    stats = file.get_stats()
                except (Exception, Warning) as err:
                    log.append("{}: {}. Ignoring".format(
                        filename, str(err).partition('.')[0]))
                    continue

            nice_name = splitext(basename(filename))[0]

            data_raw.append(stats)
            file_names.append(nice_name)

        if not len(data_raw):
            print("No single file had valid format.")
            return
        # log data
        print("{}/{} files successfully read.".format(len(files) - len(log), len(files)))
        for log_entry in log:
            print(log_entry)

        data = pd.DataFrame(data_raw, index=file_names)
        data = data.fillna(0)
        data.to_json(join('./', 'analysis.json'), orient='table')

    data = data.sort_values('weighted_vel')
    # add to remove need for normalization of index
    data['name'] = data.index

    show_data(
        data,
        args.path,
        args.detailed,
        args.space,
        args.custom)
    exit(0)
