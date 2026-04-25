#!/bin/sh

# Clean up everything beforehand
rm kaka.*

# Generate PDDL Files
./x.exe -mode gen

# Call Fast Downward
~/git/downward/fast-downward.py --plan-file kaka.sas_plan kaka.domain.pddl kaka.problem.pddl --search "astar(lmcut())"
#sed -i 's/l_gripper/L_gripper/g' kaka.sas_plan
#sed -i 's/r_gripper/R_gripper/g' kaka.sas_plan


# Read plan, test applicability
./x.exe -mode read


