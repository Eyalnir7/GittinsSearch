#!/bin/bash
# $1 - max number of diverse plans to generate
# $2 - use minimal conflict extraction (true/false)

# Clean up everything beforehand
rm kaka.*
rm found_plans/*
rm sas_plan.*.komo.log
rm -fr z.*

# Generate PDDL Files
./x.exe -mode gen




# Generate first plan
~/git/forbiditerative/plan.py --planner topk_prefix --domain kaka.domain.pddl --problem kaka.problem.pddl --number-of-plans 1 --use-local-folder > planner.log
#~/git/forbiditerative/fast-downward.py --build release64  kaka.domain.pddl  kaka.problem.pddl --search "forbid_iterative(reformulate = FORBID_SINGLE_PLAN_PREFIX, dump=false, number_of_plans=1, change_operator_names=true, external_plan_file=prefix)"
current_plan=1


i=1
while [ $i -ne $1 ]
do
    echo Checking plan $x;
    # Compare plan to previously found min conflicts
    #grep "min conflict" sas_plan.*.komo.log > min_conflicts.txt
    #python filter_plan_by_prefix.py found_plans/sas_plan.$i min_conflicts.txt > check_conflict.txt
    #grep "Conflict: not found" check_conflict.txt
    #if [ $? == 0 ]; then
    #    echo "no conflict found, proceeding to check feasibility" ;

    # Read plan, test applicability
    ./x.exe -mode read -planFile found_plans/done/sas_plan.$i -findMinConflict $2 > sas_plan.$i.komo.log
    grep 'Final verdict - feasible: 1' sas_plan.$i.komo.log
    if [ $? == 0 ]; then
       echo "found feasible solution: $i" ;
       break;
    fi
    #fi
    i=$(($i+1))
    ~/git/forbiditerative/plan.py --planner topk_prefix --domain kaka.domain.pddl --problem kaka.problem.pddl --number-of-plans $i --use-local-folder  >> planner.log  
    #~/git/forbiditerative/fast-downward.py --build release64  kaka.domain.pddl  kaka.problem.pddl --search "forbid_iterative(reformulate = FORBID_SINGLE_PLAN_PREFIX, dump=false, number_of_plans=1, change_operator_names=true, external_plan_file=prefix)" > planner.log
done

