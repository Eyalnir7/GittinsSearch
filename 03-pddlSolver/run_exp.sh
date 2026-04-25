#!/bin/bash
# $1 - cfg file
# $2 - timeout

mkdir -p exp
CURRENT_DIR=`pwd`
ROOT="$(dirname "$CURRENT_DIR")"
set -x # bash will print all the commands

for seed in 0 1 2 3 4 5 6 7 8 9; do
    for conflict in "eager" "lazy" "mr"; do
        N=1
        gamma=1.0

        base_path=exp/goal_$1/N_$N/gamma_$gamma/seed_$seed;
        mkdir -p $base_path/$conflict;    
        cp $1 $base_path/$conflict
        
        cd $base_path/$conflict; python3 $ROOT/03-pddlSolver/generate_diverse_prefix.py --LGPExecutable $ROOT/03-pddlSolver/x.exe --ForbidIterativeExecutable $ROOT/forbiditerative/fast-downward.py --N $N --conflict $conflict --timeout $2 --gamma $gamma --seed $seed --config $1 > res.csv

        for N in 2 4 8; do        
            for gamma in 1.0 0.9; do            
                base_path=exp/goal_$1/N_$N/gamma_$gamma/seed_$seed;

                mkdir -p $base_path/$conflict;
                cp $1 $base_path/$conflict

                cd $base_path/$conflict; python3 $ROOT/03-pddlSolver/generate_diverse_prefix.py --LGPExecutable $ROOT/03-pddlSolver/x.exe --ForbidIterativeExecutable $ROOT/forbiditerative/fast-downward.py --N $N --conflict $conflict --timeout $2 --gamma $gamma --seed $seed --config $1 > res.csv
            done
            base_path=exp/goal_$1/N_$N/novelty/seed_$seed;

            mkdir -p $base_path/$conflict;
            cp $1 $base_path/$conflict

            cd $base_path/$conflict; python3 $ROOT/03-pddlSolver/generate_diverse_prefix.py --LGPExecutable $ROOT/03-pddlSolver/x.exe --ForbidIterativeExecutable $ROOT/forbiditerative/fast-downward.py --N $N --conflict $conflict --timeout $2 --novelty --seed $seed --config $1 > res.csv
        done
    done

    base_path=exp/goal_$1/LGP/BFS/seed_$seed/;
    for bound in 0 1 2; do
        mkdir -p $base_path/bound_$bound;
        cp $1 $base_path/bound_$bound;    
        cd $base_path/bound_$bound; $ROOT/03-pddlSolver/x.exe -mode lgp -bound $bound -steps 1000000 -cfg $1 > output.txt; grep '#solutions' output.txt | tail -1 | gawk --assign=bound=$bound '{OFS=\"\t\"; print \"LGP\", \"BFS\", bound, \$6, \$8, \$10, \$12, \$14, \$2, \$4}' > res.csv;
    done
done





