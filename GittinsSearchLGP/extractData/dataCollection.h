#ifndef DATA_COLLECTION_H
#define DATA_COLLECTION_H

#include <LGP/LGP_TAMP_Abstraction.h>
#include <LGP/LGP_computers2.h>
#include <Core/util.h>

typedef rai::Array<rai::String> StringA;

// Solve LGP path optimization for a complete RRT path
bool solveLGPPathOptimization(
    std::shared_ptr<rai::LGPComp2_RRTpath> rrt_node,
    rai::Array<std::shared_ptr<rai::LGPComp2_RRTpath>>& rrt_nodes,
    uint lgp_trial,
    const rai::Array<StringA>& actionSequence,
    int planID,
    rai::FileToken& descFileLGP,
    std::shared_ptr<rai::LGPComp2_Skeleton> sket_node_keepalive,
    std::shared_ptr<rai::LGPComp2_Waypoints> ways_node_keepalive);

// Solve RRT for all segments with multiple seed trials
bool solveRRTSegments(
    std::shared_ptr<rai::LGPComp2_Waypoints> ways_node,
    std::shared_ptr<rai::LGPComp2_Skeleton> sket_node,
    uint T,
    int num_seed_trials,
    const rai::Array<StringA>& actionSequence,
    int planID,
    rai::FileToken& descFileRRT,
    rai::FileToken& descFileLGP);

// Solve waypoints for a skeleton and attempt RRT+LGP optimization
bool solveWaypointsForSkeleton(
    std::shared_ptr<rai::LGPComp2_Skeleton> sket_node,
    int num_seed_trials,
    int planID,
    rai::FileToken& descFileWaypoints,
    rai::FileToken& descFileRRT,
    rai::FileToken& descFileLGP,
    rai::Array<double>& times_waypoints,
    rai::Array<int>& feasibilities_waypoints,
    int max_waypoints_tries = 500);

// Main data collection function
void collectData(
    rai::LGPComp2_root& root, 
    int num_seed_trials, 
    int problemID, 
    int num_plans, 
    int verbose,
    const rai::String& dataPath,
    rai::Rnd& rnd,
    bool single_config = false,
    int max_waypoints_tries = 500,
    int max_plans = 50);

#endif // DATA_COLLECTION_H
