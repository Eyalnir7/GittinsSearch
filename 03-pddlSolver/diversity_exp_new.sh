#!/bin/bash
# $1 - number of plans to generate

N=100

if [ "$#" -gt 0 ]; then
    N=$1
fi

# Generate PDDL Files
echo "*******Generating PDDL files"
./x.exe -mode gen


# Generate set of $N plans
echo "******* Calling top-K planner to generate the first set of $N plans"
#../forbiditerative/plan.py --planner topk_prefix --domain kaka.domain.pddl --problem kaka.problem.pddl --number-of-plans $N --use-local-folder >> planner.log
../forbiditerative/plan.py --planner diverse --domain kaka.domain.pddl --problem kaka.problem.pddl --number-of-plans $N --use-local-folder >> planner.log


cp empty_cache.json cache.json

for plan in found_plans/sas_plan.*[13579]; do
    ./x.exe -mode read -planFile $plan -minConflictGapThreshold 0 > $plan.komo.log
done

python3 new_diversity_exp.py

for plan in found_plans/sas_plan.*[02468]; do
    ./x.exe -mode read -planFile $plan -minConflictGapThreshold 9999 > $plan.feas
done

