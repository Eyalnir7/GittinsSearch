#!/bin/bash
# $1 - timeout

cfg_files=`python3 generate_cfg_files.py $1`

for cfg in $cfg_files; do
    ./run_exp.sh $cfg $1
done

