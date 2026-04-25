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

num_plans=`find found_plans -name sas_plan.\* | wc -l`

echo -n "" > check_feasbility.txt
for plan in found_plans/sas_plan.*; do
    echo "./x.exe -mode read -planFile $plan -minConflictGapThreshold 9999 -useDDMDP false > $plan.komo.log" >> check_feasbility.txt
done
cat check_feasbility.txt | parallel -j `nproc`



# Compute similarity scores
echo "**** computing similarity score for $num_plans plans"
../DiverseScore/fast-downward.py kaka.domain.pddl kaka.problem.pddl --diversity-score  "score(compute_stability_metric=true,aggregator_metric=avg,plans_as_multisets=false,discounted_prefixes=true, discount_factor=0.9,dump_pairs=true)" --internal-plan-files-path found_plans --internal-num-plans-to-read $num_plans > scores.txt

python3 test_diversity_exp.py

