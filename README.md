# HADES-R

## Overview

eartHquake locAtion via Distance gEometry Solvers - RELATIVE

HADES-R (HADES - Relative) is an advanced seismic data processing tool that utilizes quaternion rotations to analyze seismic event locations in a relative sense. Designed to work with one master event, HADES-R improves upon traditional method HADES (https://github.com/wulwife/HADES) by incorporating modern computational techniques to enhance precision and efficiency.

See preprint from Tuinstra et al., 2023: 
https://doi.org/10.48550/arXiv.2309.16317


## Features

- **Quaternion Rotations**: Implements quaternion-based calculations to manage 3D rotations and transformations.


## Prerequisites

Before installing HADES-R, ensure you have the following software installed:

- **Python 3.8+**: Required for all Python dependencies.
- **Conda Environment**: Recommended for managing library dependencies.

## Dependencies

HADES-R requires the following libraries:

- `scipy>=1.9.3`
- `numpy`
- `matplotlib`
- `pandas`
- `datetime`
- `halo`
- `math`
- `time`
- `e13tools`

Additional tools:

- **latlon2cart**: Necessary for coordinate conversions. Available at [latlon2cart GitHub repository](https://github.com/wulwife/latlon2cart).
- **DAS_PICKS**: Recommended for users needing manual or semi-manual data picking. Visit [DAS_PICKS GitHub repository](https://github.com/ktuin/DAS_PICKS).

## Installation

To install HADES-R and all dependencies, follow these steps:

```bash
# Create a new Conda environment (optional but recommended)
conda create --name hades-r python=3.8
conda activate hades-r

# Install required libraries
conda install numpy scipy matplotlib pandas halo datetime time e13tools

# If some packages need to be installed via pip
pip install <package-name>
```

Ensure you have installed `latlon2cart` and `DAS_PICKS` if required by your workflow.

## Usage

See the docs!

## Contributing

We welcome contributions to HADES-R! If you would like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

Ensure to update tests as appropriate.

## License

Distributed under the GNU GENERAL PUBLIC LICENSE License. See `LICENSE` for more information.

## Contact

For any questions or feedback, please open an issue in the GitHub repository, or directly reach out via email.

---
