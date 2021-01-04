from setuptools import setup, find_packages

with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")

setup(
    name="midi_lens",
    packages=["midi_lens"],
    include_package_data=True,
    entry_points={
        "console_scripts": ['midi-lens = midi_lens.__main__:main']
    },
    url='https://github.com/J1939G/midi-lens',
    version='1.0.1',
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    description="CLI tool for visualization of MIDI files",
    long_description=long_descr,
    long_description_content_type='text/markdown',
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
