#!/bin/bash
# Run script for pr2_onTrayObstacles solver
# This sets HDF5_DISABLE_VERSION_CHECK to suppress HDF5 version mismatch warnings

export HDF5_DISABLE_VERSION_CHECK=1
./x.exe "$@"
