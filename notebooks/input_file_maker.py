# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3.8.5 ('obspy')
#     language: python
#     name: python3
# ---

# # Input file generator for HADES
#
# This file is an example of how to load and structure your data such that it can be used by HADES.

# +
# to set the path for importing HADES modules
import sys
sys.path.append("../source/")

# Import packages
import numpy as np
import pandas as pd

from hades_utils import make_datfile
from hades_utils import make_statfile
# -

# ### You need:
#
# - Station names, station coordinates (lat, lon, depth) (WGS84)
#
# - Your seismic events from ***one cluster at a time*** with:
#     - Arrival time picks (P and S) in format `%yy/%dd/%mm %H:%M:%S.%f` *for each station and each event*. All events must be picked for both P and S at the all stations. If that's not the case select only the subset of events or stations that does satisfy this criterium.
#     
#
# - Master events of choice. You must have at least ***one*** master event with an absolute location.
# - Define the ***origin*** of your cluster. Mostly this is chosen as the first master event.
#
#
# > If you have <4 master events, you should flag a total of 4 events as master event if you have just one pre-defined location. You may leave the extra flagged events locationless. E.g. you have 1 master event, then you add its location to the input file, and flag 3 other events with different $t_s - t_p$ to ensure that they don't lie close together. If you have a minimum of 4 master events you should flag the master events you want to use. 
#
#
#
# > You can make a station file with more station locations even if they are not all used in your dat file. In this way, you can run multiple scripts with one station file and linking different station combinations. You can also use the same event file with multiple station measurements and only use a subset!

# +
# Import your data

# station(s)



# master event(s) and location(s)
# !!! Make sure your depth is in km's and your coordinates in lat, lon!



# P- and S- traveltime picks datetime format (%yy/%dd/%mm %H:%M:%S.%f)





# +
# make a station file
make_statfile()

# make a dat file
make_datfile()


# +
# Quality check:

# import your datfile

# import your statfile

# -

#
