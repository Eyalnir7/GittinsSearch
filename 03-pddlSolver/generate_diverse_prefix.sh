#!/bin/bash
# $1 - gap threshold between lower/uppder bound during binary search (0 = find minimal conflict, 1000 = don't search)
# $2 - number of plans to generate per call to diverse planner (100)
# $3 - number of diverse plans to get every iteration (10)
# $4 - discount factor for plan diversity (0.0 - 1.0), where 1.0 means no discount...
# $5 - use data driven MDP for metareasoning (true/false)
# $6 - use diversity from feasible/infeasible prefixes (true) or diversity from plans (false)

minConflictGapThreshold=0
N=100
K=10
iteration=1
discounted_prefixes=false
discounted_prefixes_factor=1.0
useDDMDP=false
useDiversityFromPrefixes=false

if [ "$#" -gt 0 ]; then
    minConflictGapThreshold=$1
fi

if [ "$#" -gt 1 ]; then
    N=$2
fi

if [ "$#" -gt 2 ]; then
    K=$3
fi

if [ "$#" -gt 3 ]; then
    discounted_prefixes_factor=$4
    if (( $(echo "$4 < 1" |bc -l) )) ; then
        discounted_prefixes=true;
    fi    
fi

if [ "$#" -gt 4 ]; then
    useDDMDP=$5
fi

if [ "$#" -gt 5 ]; then
    useDiversityFromPrefixes=$6
    echo "useDiversityFromPrefixes = $useDiversityFromPrefixes"
fi

# Clean up everything beforehand
rm kaka.*
rm found_plans/*
rm sas_plan.*
rm -fr z.*
rm planner.log reformulate.log diversescore.log



# create empty cache
cp empty_cache.json cache.json

# found_plans/done/sas_plan.$i - either full original plan or conflict, depending on where
# found_plans/done/sas_plan.$i.full - original plan
# found_plans/done/sas_plan.$i.conflict - conflict extracted from a tested plan


# Generate PDDL Files
echo "*******Generating PDDL files"
./x.exe -mode gen


# Generate first set of $N plans
echo "******* Calling top-K planner to generate the first set of $N plans"
../forbiditerative/plan.py --planner topk_prefix --domain kaka.domain.pddl --problem kaka.problem.pddl --number-of-plans $N --use-local-folder >> planner.log

tested_num=1
SOLVED=0
while [ $SOLVED -ne 1 ] 
do
    num_plans=$(( N * iteration ))
    num_tested_plans=$(( K * (iteration - 1) ))
    num_total_plans_to_test=$(( K * iteration ))
    echo "******* Expecting $num_plans plans"
    # Choose diverse subset of size $K
    echo "******* Choosing diverse subset of size $K  (num_plans = $num_plans, num_tested_plans=$num_tested_plans, num_total_plans_to_test=$num_total_plans_to_test)"
    
    if [ "$useDiversityFromPrefixes" == "true" ]; then
        echo "**** Using diversity from prefixes"
        python3 choose_diverse.py --plansdir found_plans/done --num-plans-to-skip $num_tested_plans > d.log;
    else
        echo "**** Using diversity from plans"
        python3 ../diversescore/fast-downward.py kaka.domain.pddl kaka.problem.pddl --diversity-score "subset(compute_stability_metric=true,aggregator_metric=avg,plans_as_multisets=false,plans_subset_size=$num_total_plans_to_test,exact_method=false,dump_plans=true,plans_seed_set_size=$num_tested_plans,discounted_prefixes=$discounted_prefixes, discount_factor=$discounted_prefixes_factor)" --internal-plan-files-path found_plans/done --internal-num-plans-to-read $num_plans > d.log;
    fi

    cat d.log >> diversescore.log
    # get chosen plan indices, sorted by plan index (so the plan shuffling in the next step will work correctly)
    new_chosen_plan_indices=`grep 'Plan index: ' d.log | tail -$K | gawk '{print $NF+1;}' | sort -n`

    echo "*********** chosen: $new_chosen_plan_indices"


    # Process chosen set of plans
    echo "******* Testing chosen plans"
    for chosen_plan_index in $new_chosen_plan_indices; do
        #swap chosen plan with next untested plan, to maintain that the first $num_tested plans have been tested
        echo "Swapping plan $chosen_plan_index with plan $tested_num (if needed)"
        if [ $chosen_plan_index -ne $tested_num ]; then             
            mv found_plans/done/sas_plan.$tested_num sas_plan.tmp
            mv found_plans/done/sas_plan.$chosen_plan_index found_plans/done/sas_plan.$tested_num
            mv sas_plan.tmp found_plans/done/sas_plan.$chosen_plan_index
        fi
        
        echo "******* Testing plan found_plans/done/sas_plan.$tested_num"        
        cat found_plans/done/sas_plan.$tested_num
        ./x.exe -mode read -planFile found_plans/done/sas_plan.$tested_num -minConflictGapThreshold $minConflictGapThreshold -useDDMDP $useDDMDP > found_plans/done/komo.log.sas_plan.$tested_num
        grep 'Final verdict - feasible: 1' found_plans/done/komo.log.sas_plan.$tested_num
        if [ $? == 0 ]; then
            echo "******* found feasible solution:found_plans/done/sas_plan.$tested_num" ;
            SOLVED=1
            exit;
        fi        
        
        grep 'min conflict:' found_plans/done/komo.log.sas_plan.$tested_num | gawk 'BEGIN {FS="(";} {for (i=2; i <= NF; i++) {gsub(/^[ \t]+|[ \t]+$/, "",$i); print "(" $i;}}' > found_plans/done/sas_plan.$tested_num.conflict
                
        tested_num=$(( tested_num + 1 ))
    done


    iteration=$(( iteration + 1))
    echo "******* Starting iteration $iteration"
    
    # rename conflicts to be used for forbidding compilation, next
    echo "******* Renaming conflicts"
    for conflict in found_plans/done/sas_plan.*.conflict; do 
        plan=${conflict%.conflict}
        mv $plan $plan.full
        mv $conflict $plan
    done


    I=0
    while [ $I -lt $N ] 
    do
        echo "******* Reformulating to forbid found plans and conflicts ($I / $N)"
        temp_num_plans=$(( num_plans + I ))
        next_plan_num=$(( temp_num_plans + 1 ))
        
        ../forbiditerative/fast-downward.py --build release64  kaka.domain.pddl  kaka.problem.pddl --search "forbid_iterative(reformulate = FORBID_MULTIPLE_PLAN_PREFIXES, change_operator_names=false,number_of_plans_to_read=$temp_num_plans, external_plans_path=found_plans/done/)" >> reformulate.log
        
        echo "******* Calling planner ($I / $N)"
        ../forbiditerative/fast-downward.py --build release64 reformulated_output.sas  --symmetries "sym=structural_symmetries(time_bound=0,search_symmetries=oss, stabilize_initial_state=false, keep_operator_symmetries=false)" --search "astar(celmcut(), shortest=false, symmetries=sym)" >> planner.log


        echo "******* Moving sas_plan to found_plans/done"
        mv sas_plan found_plans/done/sas_plan.$next_plan_num
        

        I=$(( I + 1 ))
    done          

    # rename conflicts back, to be used for diverse selection, in next iteration
    echo "******* Renaming plans"
    for full in found_plans/done/sas_plan.*.full; do 
        plan=${full%.full}
        mv $plan $plan.conflict
        mv $full $plan
    done
    


done


