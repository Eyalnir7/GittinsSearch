#include "dataCollection.h"
#include <Optim/NLP_Solver.h>
#include <PathAlgos/RRT_PathFinder.h>
#include <Kin/frame.h>
#include <LGP/NLP_Descriptor.h>
#include <Kin/viewer.h>
#include <iostream>

//===========================================================================
// IMPORTANT MEMORY SAFETY NOTE:
// The LGP computation tree uses raw pointers internally for parent references:
// - LGPComp2_OptimizePath (path_node) stores raw pointer: LGPComp2_Skeleton* sket
// - LGPComp2_Skeleton stores raw pointer: LGPComp2_root* root
// - During path_node->compute(), it accesses sket->root (raw pointer chain)
//
// To prevent dangling pointers and segfaults:
// 1. Keep the root alive in main() as shared_ptr throughout collectData()
// 2. Keep skeleton node alive as shared_ptr throughout waypoints/RRT/path optimization
// 3. Keep waypoints node alive as shared_ptr throughout RRT/path optimization
// 4. Never clear children or reset shared_ptrs while nodes are computing
//===========================================================================

// Solve LGP path optimization for a complete RRT path
bool solveLGPPathOptimization(
    std::shared_ptr<rai::LGPComp2_RRTpath> rrt_node,
    rai::Array<std::shared_ptr<rai::LGPComp2_RRTpath>>& rrt_nodes,
    uint lgp_trial,
    const rai::Array<StringA>& actionSequence,
    int planID,
    rai::FileToken& descFileLGP,
    std::shared_ptr<rai::LGPComp2_Skeleton> sket_node_keepalive,
    std::shared_ptr<rai::LGPComp2_Waypoints> ways_node_keepalive)
{
  cout << "[DEBUG] All RRT segments feasible, creating path optimization node" << endl;
  
  // Safety check: ensure skeleton and root are still valid (they should be kept alive by shared_ptrs)
  if(!sket_node_keepalive || !ways_node_keepalive) {
    cout << "[SAFETY] Skeleton or waypoints node was deleted - cannot create path node" << endl;
    return false;
  }
  
  auto path_node = std::dynamic_pointer_cast<rai::LGPComp2_OptimizePath>(
    rrt_node->createNewChild(lgp_trial));
  if(!path_node) {
    cout << "Failed to create path optimization node" << endl;
    return false;
  }
  
  cout << "[DEBUG] Computing path optimization" << endl;
  while(!path_node->isComplete) {
    if(!path_node || !sket_node_keepalive || !ways_node_keepalive) {
      cout << "[ERROR] Node became null during path optimization compute" << endl;
      // return false;
    }
    cout << "Computing path optimization..." << endl;
    path_node->compute();
  }
  cout << "[DEBUG] Path optimization complete" << endl;
  double time_path = path_node->c;
  
  descFileLGP << path_node->isFeasible << "#" << time_path << "#" << actionSequence << "#" << planID << endl;
  return true;
}

// Solve RRT for all segments with multiple seed trials
bool solveRRTSegments(
    std::shared_ptr<rai::LGPComp2_Waypoints> ways_node,
    std::shared_ptr<rai::LGPComp2_Skeleton> sket_node,
    uint T,
    int num_seed_trials,
    const rai::Array<StringA>& actionSequence,
    int planID,
    rai::FileToken& descFileRRT,
    rai::FileToken& descFileLGP)
{
  cout << "[DEBUG] Getting number of segments T=" << T << ", Running " << num_seed_trials << " RRT trials with different seeds..." << endl;
  int lgp_trial = 0;
  int rrt_trial = 0;
  rai::Array<rai::Array<double>> times_rrts;
  rai::Array<rai::Array<int>> feasibilities_rrts;
  rai::Array<arr> q0s_rrts;
  rai::Array<arr> qfs_rrts;
  
  // Initialize the arrays to have T array elements
  times_rrts.resize(T);
  feasibilities_rrts.resize(T);
  q0s_rrts.resize(T);
  qfs_rrts.resize(T);
  for (uint t = 0; t < T; t++) {
    times_rrts(t).clear();
    feasibilities_rrts(t).clear();
  }
  
  while(rrt_trial < num_seed_trials) {
    cout << "[DEBUG] Starting LGP trial " << lgp_trial << "/" << num_seed_trials << endl;
    bool all_rrt_feasible = true;
    std::shared_ptr<rai::LGPComp2_RRTpath> rrt_node = nullptr;
    rai::Array<std::shared_ptr<rai::LGPComp2_RRTpath>> rrt_nodes(T);
    uint total_path_nodes = 0;
    // rai::Array<double> concat_path;  // Collect path data immediately
    rrt_trial++;
    
    for (uint t = 0; t < T; t++)
    {
      cout << "[DEBUG] Processing RRT segment " << t << "/" << T << endl;
      // Create RRT node for this segment with seed based on trial number
      if(t == 0) {
        cout << "[DEBUG] Creating first RRT node from waypoints" << endl;
        rrt_node = std::dynamic_pointer_cast<rai::LGPComp2_RRTpath>(
          ways_node->createNewChild(rrt_trial));
        cout << "Created RRT node from waypoint for segment " << t << endl;
      } else {
        rrt_node = std::dynamic_pointer_cast<rai::LGPComp2_RRTpath>(
          rrt_node->createNewChild(rrt_trial));
        cout << "Created RRT node from previous for segment " << t << endl;
      }
      
      if(!rrt_node) {
        cout << "Failed to create RRT node for segment " << t << endl;
        all_rrt_feasible = false;
        break;
      }

      if(rrt_trial == 1){
        q0s_rrts(t) = rrt_node->q0;
        qfs_rrts(t) = rrt_node->qT;
      }
      
      rrt_nodes(t) = rrt_node;
      
      // Compute RRT until complete
      while(!rrt_node->isComplete) {
        if(!rrt_node) {
          cout << "[ERROR] rrt_node became null during compute for segment " << t << endl;
        }
        rrt_node->compute();
        cout << "Computing RRT for segment " << t << endl;
      }
      cout << "RRT for segment " << t << " completed." << endl;
      double time_rrt = rrt_node->c;
      times_rrts(t).append(time_rrt);
      if (!rrt_node->isFeasible)
      {
        cout << "RRT for segment " << t << " not feasible" << endl;
        feasibilities_rrts(t).append(0);
        all_rrt_feasible = false;
        break;
      }
      else {
        cout << "RRT for segment " << t << " feasible" << endl;
        feasibilities_rrts(t).append(1);
      }
      
      // Collect path data immediately - use path member, not rrt (which gets reset to null)
      // if(rrt_node && rrt_node->path.N > 0) {
      //   for(uint k = 0; k < rrt_node->path.d0; k++) {
      //     for(uint j = 0; j < rrt_node->path.d1; j++) {
      //       concat_path.append(rrt_node->path(k, j));
      //     }
      //   }
      //   total_path_nodes += rrt_node->path.d0;
      // }
    }

    if(!all_rrt_feasible){
      cout << "[DEBUG] Not all RRT segments feasible for LGP trial " << lgp_trial << ", skipping LGP optimization." << endl;
      for(uint t = 0; t < rrt_nodes.N; t++) {
        if(rrt_nodes(t)) rrt_nodes(t)->children.clear();
      }
      cout << "[DEBUG] Cleared children of all RRT nodes for LGP trial " << lgp_trial << endl;
      rrt_nodes.clear();
      // concat_path.clear();
      continue;
    }
    
    lgp_trial++;
    
    // Solve LGP path optimization (pass shared_ptrs to keep skeleton and waypoints alive)
    solveLGPPathOptimization(rrt_node, rrt_nodes, lgp_trial, actionSequence, planID, 
                             descFileLGP, sket_node, ways_node);
    
    // Clear collected paths to free memory
    // concat_path.clear();
    
    // Clear children of all rrt_nodes after this trial
    for(uint t = 0; t < rrt_nodes.N; t++) {
      if(rrt_nodes(t)) rrt_nodes(t)->children.clear();
    }
    rrt_nodes.clear();
  }
  
  // Log RRT results for all segments
  for (uint t = 0; t < T; t++) {
    descFileRRT << feasibilities_rrts(t) << "#" << times_rrts(t) << "#" << actionSequence << "#" << planID << "#" << t << "#" << q0s_rrts(t) << "#" << qfs_rrts(t) << endl;
  }
  
  // Clear RRT arrays
  times_rrts.clear();
  feasibilities_rrts.clear();
  q0s_rrts.clear();
  qfs_rrts.clear();
  
  return true;
}

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
    int max_waypoints_tries)
{
  uint num_tries = 0;
  
  while(num_tries < (uint)max_waypoints_tries)
  {
    num_tries++;
    
    // Print progress every 50 tries
    if(num_tries > 0 && num_tries % 50 == 0) {
      cout << "Progress: " << num_tries << " tries" << endl;
    }
    
    // Create waypoints node with random seed
    cout << "[DEBUG] Creating waypoints node, try " << num_tries << endl;
    auto ways_node = std::dynamic_pointer_cast<rai::LGPComp2_Waypoints>(
      sket_node->createNewChild(num_tries));
    if(!ways_node) {
      cout << "Failed to create waypoints node" << endl;
      continue;
    }
    cout << "[DEBUG] Waypoints node created successfully" << endl;
    
    cout << "[DEBUG] Computing waypoints node" << endl;
    while(!ways_node->isComplete) {
      ways_node->compute();
    }
    cout << "[DEBUG] Waypoints computation complete" << endl;
    double time_waypoints = ways_node->c;
    times_waypoints.append(time_waypoints);
    
    if (ways_node->isFeasible)
    {
      cout << "Found feasible waypoint solution at try " << num_tries << endl;
      feasibilities_waypoints.append(1);
    }
    else {
      feasibilities_waypoints.append(0);
      ways_node->children.clear();
      // DON'T clear skeleton children here - we're still in the waypoints retry loop
      // and may need the skeleton structure intact for the next waypoint attempt
      continue;
    }

    // Solve RRT for each segment
    cout << "[DEBUG] Getting number of segments T" << endl;
    uint T = ways_node->komoWaypoints->T;
    
    // Pass skeleton shared_ptr to keep it alive during RRT and LGP optimization
    solveRRTSegments(ways_node, sket_node, T, num_seed_trials, sket_node->actionSequence, planID,
                     descFileRRT, descFileLGP);
    
    // Clear waypoints children after all RRT trials complete
    if(ways_node) ways_node->children.clear();
  }
  
  // Clear skeleton children only once, after all waypoint attempts
  if(sket_node) sket_node->children.clear();
  
  return true;
}

void collectData(
    rai::LGPComp2_root& root, 
    int num_seed_trials, 
    int problemID, 
    int num_plans, 
    int verbose,
    const rai::String& dataPath,
    rai::Rnd& rnd,
    bool single_config,
    int max_waypoints_tries,
    int max_plans)
{
  cout << "[DEBUG] Entering collectData for problemID=" << problemID << endl;
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////// INITIALIZATION ////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  rai::setParameter<int>("KOMO/verbose", verbose - 2);
  
  // Always create file descriptors
  rai::FileToken descFileWaypoints(FILE(STRING(dataPath + "waypoints/z.dataWaypoints" << problemID)));
  descFileWaypoints << "feas#time#plan#planID" << endl;
  rai::FileToken descFileRRT(FILE(STRING(dataPath + "rrt/z.dataRRT" << problemID)));
  descFileRRT << "feas#time#plan#planID#actionNum#q0#qf" << endl;
  rai::FileToken descFileLGP(FILE(STRING(dataPath + "lgp/z.dataLGP" << problemID)));
  descFileLGP << "feas#time#plan#planID#RRTPath" << endl;
  int countFeas = 0;
  int countNotFeas = 0;
  double avgtimeFeas = 0;
  double avgtimeNotFeas = 0;
  rai::Array<double> times_waypoints;
  rai::Array<int> feasibilities_waypoints;

  cout << "[DEBUG] Initialization complete, creating sampled numbers" << endl;
  rai::Array<int> sampledNums;
  if(single_config) {
    // For single config, just use sequential numbers 1 to num_plans
    for(int b = 0; b < num_plans; b++) {
      sampledNums.append(b + 1);
    }
  } else {
    // For random configs, sample randomly
    for(int b = 0; b < num_plans; b++) {
      if(num_plans >= max_plans){
        sampledNums.append(b + 1);
        continue;
      }
      sampledNums.append(rnd.uni_int(1, max_plans));
    }
  }
  cout << "[DEBUG] Sampled numbers: " << sampledNums << endl; 

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////// MAIN LOOP /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  for (int i = 0; i < max_plans; i++)
  {
    cout << "[DEBUG] Main loop iteration i=" << i << endl;
    if (!sampledNums.contains(i + 1)){
      auto taskPlan = root.tamp.getNewActionSequence();
      cout << "[DEBUG] Skipping plan " << i << " (not in sampled nums)" << endl;
      cout << "action sequence: " << taskPlan << endl;
      continue;
    }
    // Create skeleton node to get action sequence
    cout << "[DEBUG] Creating skeleton node for plan " << i << endl;
    auto sket_node = std::dynamic_pointer_cast<rai::LGPComp2_Skeleton>(root.createNewChild(i));
    if(!sket_node) {
      cout << "Failed to create skeleton node " << i << endl;
      continue;
    }
    cout << "[DEBUG] Computing skeleton node " << i << endl;
    sket_node->compute(); // find actions sequence
    cout << "[DEBUG] Skeleton computation complete" << endl;
    
    if(!sket_node->isComplete || !sket_node->isFeasible) {
      cout << "Skeleton node " << i << " failed or incomplete" << endl;
      continue;
    }
  
    cout<< "plan number " << i << ", action sequence: " << sket_node->actionSequence << endl;

    // Solve waypoints, RRT, and LGP path optimization for this skeleton
    solveWaypointsForSkeleton(sket_node, num_seed_trials, i,
                              descFileWaypoints, descFileRRT, descFileLGP,
                              times_waypoints, feasibilities_waypoints,
                              max_waypoints_tries);

    cout << "[DEBUG] Clearing skeleton node children" << endl;
    sket_node->children.clear();
    
    cout << "[DEBUG] Plan " << i << " completed: seed effect analysis done" << endl;

    // Log the waypoints results
    descFileWaypoints << feasibilities_waypoints << "#" << times_waypoints << "#" << sket_node->actionSequence << "#" << i << endl;
    // clear waypoints arrays
    times_waypoints.clear();
    feasibilities_waypoints.clear();
  }
  
  cout << "[DEBUG] Exiting collectData" << endl;
}
