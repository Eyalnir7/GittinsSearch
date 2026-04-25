#!/bin/bash

echo -e config'\t'N'\t'conflict'\t'gamma'\t'novelty'\t'seed'\t'solved'\t'length'\t'generated'\t'tested'\t'pose'\t'seq'\t'path'\t'time > raw.csv

for g in exp/goal_*; do 
    #echo $g
    for x in `find $g -name res.csv | grep -v LGP`; do cat $x; done | sort >> raw.csv
done

echo config LGP BFS bound KIN TREE pose seq path time timeint > baseline.csv

for g in exp/goal_*; do 
    #echo $g
    for x in `find $g -name res.csv | grep LGP`; do echo $g `cat $x`; done | sort >> baseline.csv
done

