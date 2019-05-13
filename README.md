# Undrift
Corrects local, non-linear drift in tissue by estimating dense optical flow and
using it for warping images back to first frame.

It comes as command line tool for Python >= 3.6

### Installation
1. Clone repository to `<path>`
2. With (Ananconda) command line `cd <path>/undrift`
3. Pip install undrift with:

    ```pip install -r requirements.txt -e .```

Required python packages are installed automatically

### Usage

Basic command:

```undrift <movie_tif_file> [<more_tif_files>...]```

Command line help:

```undrift --help```

### Tips
to use undrift with spatial smoothing of 50 px for all tif files in a folder recursively on Windows, use

```for /r %i in (*.tif) do undrift --smooth_xy 50 "%i"```