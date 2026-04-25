#!/bin/sh
# $1 - number of diverse plans to generate

# Clean up everything beforehand
rm kaka.*
rm found_plans/*
rm -fr z.*

# Generate PDDL Files
./x.exe -mode gen

# Call Fast Downward
~/git/forbiditerative/plan_topk_prefix.sh kaka.domain.pddl kaka.problem.pddl $1

for x in found_plans/*; do
  echo Checking plan $x;
  #sed -i 's/l_gripper/L_gripper/g' $x;
  #sed -i 's/r_gripper/R_gripper/g' $x;
  # Read plan, test applicability
  ./x.exe -mode read -planFile $x
done


