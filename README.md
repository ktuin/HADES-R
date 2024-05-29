# HADES-R


## Overview

eartHquake locAtion via Distance gEometry Solvers - RELATIVE

HADES-R (HADES - Relative) is an advanced seismic data processing tool that utilizes quaternion rotations to analyze seismic event locations in a relative sense. Designed to work with one master event, HADES-R improves upon traditional method HADES (https://github.com/wulwife/HADES) by incorporating modern computational techniques to enhance precision and efficiency.


# References
This code is the basis for the following open-access paper, please read for more details: 

- Tuinstra, Katinka et al. "Locating clustered seismicity using Distance Geometry Solvers: applications for sparse and single-borehole DAS networks", Geophysical Journal International (2024);, ggae168, https://doi.org/10.1093/gji/ggae168

It also enables the use of classical HADES:

- Grigoli, Francesco, et al. "Relative earthquake location procedure for clustered seismicity with a single station." Geophysical Journal International 225.1 (2021): 608-626 (https://doi.org/10.1093/gji/ggaa607)

# Notes
Please contact us for collaborations and/or suggestions!

--> See examples for an input file maker and a synthetic example
--> See docs html for more information on code use
--> It is possible to input inter-event distances calculated with an external method (cross-correlations, neural networks, etc.)

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
