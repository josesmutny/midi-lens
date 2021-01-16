# MIDI file analyzer and comparer

  - [How to...](#how-to)
    - [Install](#install)
    - [Use](#use)
    - [Understand visualizations](#understand-visualizations)
  - [Analysis method](#analysis-method)
    - [Individual pieces](#individual-pieces)
    - [Batch of pieces](#batch-of-pieces)
  - [WIP and to-dos](#wip-and-to-dos)

The `midi-lens` script allows for graphical analysis and comparison of midi files. 
Batch analysis compares pieces by piece-wide metrics (avg tempo, piece length etc.) and 
single file analysis provides slice-based statistics.  
More on the [**Analysis method**](##analysis-method) section.

The results of batch analysis (which can take some time for large samples) are stored into .json files, which can be reused. More on [**How to...**](##how-to...)

## How to...

### Install 

For normal use, `poetry install` will suffice. 

For development:
1. Clone this repository.
2. Go to `midi_analyzer/`.
3. `poetry install`.
4. `poetry run python -m midi_lens`.

### Use

Open terminal. Then type:
- `midi-lens` to analyze all .mid files in current directory.
- `midi-lens path` to analyze files in `path`.
- `midi-lens path/filename` to analyze single file.

After batch analysis an analysis.json file will be produced. To reuse it, type `midi-lens analysis.json`. It can have any other name, as long as the contents are valid. 

All of the above commands can be modified using the following flags (which can be displayed from terminal using `midi-lens --help`)
- `--no-pedal` Ignore piano sustain pedal. If used, it is likely that more notes will appear, as sustain generally produces less slices (more on [methodics](##analysis-method)).
- `--detailed` Produce more detailed visualizations.
- `--space` Produce 3D visualizations.
- `--ignore-breaks` Ignore slices with no sound (silence).
- `--custom` Generate custom visualization based on user input.
- `--frequency` Determine width of time slices (higher numbers mean higher detail). Default is 48 (i.e. 48 slices per second) 

All of this can be applied simultaneously, so for instance to get a detailed analysis of a single file then type:
~~~
midi-lens path/filename --detailed --space --custom
~~~

### Understand visualizations
An attempt was made to provide clear titles and labels for graphs. A good starting point for single file analysis is to produce 2D plots where the `x` axis is `index`, and the `y` axis is any variable. This produces a graph representing the changes in the `y` axis through time, which is relatively easy to understand. If color is too distracting, it can be set to be either `x` or `y`.

Visualizations may be also good for comparing pieces, as they are good for showing the differences in execution. For example, one may realize by plotting the change of note velocity through time that he plays one whole section a lot louder than the other, as opposed to a reference file. 

## Analysis method

### Individual pieces
The data point for analysing individual pieces is a time **slice**. This is a moment of time which is distinguishable from the ones around it. Take for instance the following piece, where each line represents a note, and each column is a frame of time (by default 48 ms), represented by t. 
```
t: 0 1 2 3 4 5      t: 0 1 3
C: 0 0 0 x x x  --> C: 0 0 x
D: 0 x x x x x      D: 0 x x
```
This simple piece has 5 slices, and thus has a length of 5 * 48 ms = 240 ms (0.25 s).
The time slices are definitely more compact than the original data, and represent a moment in time when sound was not changing (the same notes were being played).

### Batch of pieces
Multiple pieces are analyzed by piece-wide metrics, which are produced by calculating the mean or standard deviation values from the analysis of individual pieces. **Ranges** are also calculated (difference between min and max values), so that two pieces with different characteristics can be compared easily. 

## WIP and to-dos

- [x] Fix `--no-pedal` tag not working. Done, the parameter was not being passed to MIDI 
- [ ] Allow for piece comparison. A `midi-lens --compare x.mid y.mid` would be nice, and not too hard to implement
- [ ] Batch piece comparison using `timelapse` property
