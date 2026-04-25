#!/bin/bash

# Find project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# ./x.exe -numObjLowerBound 2 \
#         -numObjUpperBound 2 \
#         -numGoalsUpperBound 2 \
#         -numBlockedGoalsUpperBound 2 \
#         -experimentName "fix_warmup_tuned_full" \
#         -numIterations 1 \
#         -GNN/modelsDir "$PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/tuned_scripted/" \
#         -mode "run" \
#         -solver GITTINS \
#         -predictionType GNN \
#         -dataPercentage 1.0 \
#         -GITTINS/numWaypoints 30 \
#         -numTaskPlans 5 \
#         -Bandit/beta 0.9999 \
#         -useDatasizeSubdir false

# ./x.exe -numObjLowerBound 3 \
#         -numObjUpperBound 3 \
#         -numGoalsUpperBound 3 \
#         -numBlockedGoalsUpperBound 2 \
#         -experimentName "fix_warmup_tuned_full" \
#         -numIterations 1 \
#         -GNN/modelsDir "$PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/tuned_scripted/" \
#         -mode "run" \
#         -solver GITTINS \
#         -predictionType GNN \
#         -dataPercentage 1.0 \
#         -GITTINS/numWaypoints 30 \
#         -numTaskPlans 10 \
#         -Bandit/beta 0.9999 \
#         -useDatasizeSubdir false

# ./x.exe -numObjLowerBound 4 \
#         -numObjUpperBound 4 \
#         -numGoalsUpperBound 4 \
#         -numBlockedGoalsUpperBound 1 \
#         -experimentName "fix_warmup_tuned_full" \
#         -numIterations 1 \
#         -GNN/modelsDir "$PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/tuned_scripted/" \
#         -mode "run" \
#         -solver GITTINS \
#         -predictionType GNN \
#         -dataPercentage 1.0 \
#         -GITTINS/numWaypoints 30 \
#         -numTaskPlans 5 \
#         -Bandit/beta 0.9999 \
#         -useDatasizeSubdir false

# ./x.exe -numObjLowerBound 2 \
#         -numObjUpperBound 2 \
#         -numGoalsUpperBound 2 \
#         -numBlockedGoalsUpperBound 2 \
#         -experimentName "fix_warmup_2blocksTrain" \
#         -numIterations 1 \
#         -GNN/modelsDir "$PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/rb2blocks_scripted/" \
#         -mode "run" \
#         -solver GITTINS \
#         -predictionType GNN \
#         -dataPercentage 1.0 \
#         -GITTINS/numWaypoints 30 \
#         -numTaskPlans 5 \
#         -Bandit/beta 0.9999 \
#         -useDatasizeSubdir false

# ./x.exe -numObjLowerBound 3 \
#         -numObjUpperBound 3 \
#         -numGoalsUpperBound 3 \
#         -numBlockedGoalsUpperBound 2 \
#         -experimentName "fix_warmup_2blocksTrain" \
#         -numIterations 1 \
#         -GNN/modelsDir "$PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/rb2blocks_scripted/" \
#         -mode "run" \
#         -solver GITTINS \
#         -predictionType GNN \
#         -dataPercentage 1.0 \
#         -GITTINS/numWaypoints 30 \
#         -numTaskPlans 10 \
#         -Bandit/beta 0.9999 \
#         -useDatasizeSubdir false

# ./x.exe -numObjLowerBound 4 \
#         -numObjUpperBound 4 \
#         -numGoalsUpperBound 4 \
#         -numBlockedGoalsUpperBound 1 \
#         -experimentName "fix_warmup_2blocksTrain" \
#         -numIterations 1 \
#         -GNN/modelsDir "$PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/rb2blocks_scripted/" \
#         -mode "run" \
#         -solver GITTINS \
#         -predictionType GNN \
#         -dataPercentage 1.0 \
#         -GITTINS/numWaypoints 30 \
#         -numTaskPlans 5 \
#         -Bandit/beta 0.9999 \
#         -useDatasizeSubdir false

./x.exe -numObjLowerBound 2 \
        -numObjUpperBound 2 \
        -numGoalsUpperBound 2 \
        -numBlockedGoalsUpperBound 2 \
        -experimentName "fix_warmup_2blocks3blocksTrain" \
        -numIterations 1 \
        -GNN/modelsDir "$PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/rb2blocks3blocks_scripted/" \
        -mode "run" \
        -solver GITTINS \
        -predictionType GNN \
        -dataPercentage 1.0 \
        -GITTINS/numWaypoints 30 \
        -numTaskPlans 5 \
        -Bandit/beta 0.9999 \
        -useDatasizeSubdir false

./x.exe -numObjLowerBound 3 \
        -numObjUpperBound 3 \
        -numGoalsUpperBound 3 \
        -numBlockedGoalsUpperBound 2 \
        -experimentName "fix_warmup_2blocks3blocksTrain" \
        -numIterations 1 \
        -GNN/modelsDir "$PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/rb2blocks3blocks_scripted/" \
        -mode "run" \
        -solver GITTINS \
        -predictionType GNN \
        -dataPercentage 1.0 \
        -GITTINS/numWaypoints 30 \
        -numTaskPlans 10 \
        -Bandit/beta 0.9999 \
        -useDatasizeSubdir false

./x.exe -numObjLowerBound 4 \
        -numObjUpperBound 4 \
        -numGoalsUpperBound 4 \
        -numBlockedGoalsUpperBound 1 \
        -experimentName "fix_warmup_2blocks3blocksTrain" \
        -numIterations 1 \
        -GNN/modelsDir "$PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/rb2blocks3blocks_scripted/" \
        -mode "run" \
        -solver GITTINS \
        -predictionType GNN \
        -dataPercentage 1.0 \
        -GITTINS/numWaypoints 30 \
        -numTaskPlans 5 \
        -Bandit/beta 0.9999 \
        -useDatasizeSubdir false