#!/bin/sh
# $1 - plan file name to check
# $2 - find min conflicts


cd /home/karpase/git/lgp-pddl/03-pddlSolver/
#cp $1 kaka.sas_plan
cat kaka.sas_plan
/home/karpase/git/lgp-pddl/03-pddlSolver/x.exe -mode read -planFile kaka.sas_plan -findMinConflict $2 > $1.lgp.log
