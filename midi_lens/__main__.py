"""__main__: executed when app directory is called as script."""

from .midi_lens import main

import glob
import os
from argparse import ArgumentParser


parser = ArgumentParser(
    description="""
        Analyze Midi files, and produce visualizations of tempo, pitch, lenght and velocity changes.
        Analysis findings are saved into .json file, which can be later reused for faster processing""")

parser.add_argument('path',
                    nargs='?',
                    default=os.getcwd(),
                    help="""Specify any of the following:
                        A path to a .mid file for detailed analysis.
                        A path to a directory containing .mid files to be analyzed in batch, producing an analysis.json file.
                        A path to an .json analysis file previously produced (must be non-empty and have appropriate columns""")
parser.add_argument('--no-pedal',
                    action='store_true',
                    help="""!WIP If set, analysis differentiates between two notes played apart but sounding at the same time due to sustain pedal usage.
                        Otherwhise this notes are considered to be played at the same time""")
parser.add_argument('-d', '--detailed',
                    action='store_true',
                    help="""Display more detailed statistics.
                        Selection of statistics occurs after their recollection, and thus this option does not have a high impact on performance""")
parser.add_argument('-s', '--space',
                    action='store_true',
                    help="""Display visualizations in space (3D)""")
parser.add_argument('-i', '--ignore-breaks',
                    action='store_true',
                    help="""Ignore sequences of piece without any notes being pressed (silence). May be used in conjuction with --no-pedal""")
parser.add_argument('-c', '--custom',
                    action='store_true',
                    help='Ask user for parameters for custom plots')
parser.add_argument('-f', '--frequency',
                    default=48,
                    help='Number of time slices in a second. Defaults to 48')
parser.add_argument('-x', '--compare',
                    default=None,
                    help='Compare path with second file.')

args = parser.parse_args()
main(args)
