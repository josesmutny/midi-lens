from setuptools import setup, find_packages

with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")

setup(
    name="midi_lens",
    packages=["midi_lens"],
    entry_points={
        "console_scripts": ['midi-lens = midi_lens.__main__:main']
    },
    version=0.3,
    description="Python command line application to analyze and compare MIDI files.",
    long_description=long_descr,
    author="José Smutný",
    author_email="smutnjo2@cvut.cz",
    install_requires=[
        "argparse",
        "pychord",
        "pandas",
        "plotly",
        "numpy",
        "tqdm",
        "mido",
        "pick",
    ]
)
