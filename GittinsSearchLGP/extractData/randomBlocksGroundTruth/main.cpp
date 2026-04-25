#include <LGP/LGP_TAMP_Abstraction.h>
#include <Optim/NLP_Solver.h>
#include <PathAlgos/RRT_PathFinder.h>
#include <Kin/frame.h>
#include <LGP/NLP_Descriptor.h>
#include <Kin/viewer.h>
#include "ConfigurationGenerator.h"
# include <filesystem>
#include <iostream>
#include <LGP/LGP_computers2.h>
#include <KOMO/pathTools.h>
#include <cstdlib>

// Get project root from environment or use fallback hardcoded path
rai::String getProjectRoot() {
  const char* project_root = std::getenv("PROJECT_ROOT");
  if (project_root) {
    return rai::String(project_root);
  }
  return "";  // fallback
}

rai::String dataPath = getProjectRoot() + "/GittinsSearchLGP/FolTest/data/testSeedEffect/";
rai::Rnd rnd;

//===========================================================================

void checkSeedEffect(rai::LGPComp2_root& root, int num_trials, int problemID, int verbose = 10, bool logVerbose = false)
{
  uint num_seed_trials = 100;  // 100 tries for each seed effect test
  rai::setParameter<int>("KOMO/verbose", verbose - 2);
  
  rai::FileToken* descFileWaypoints = nullptr; 
  rai::FileToken* descFileRRT = nullptr;
  rai::FileToken* descFileLGP = nullptr;
  if(logVerbose){
    descFileWaypoints = new rai::FileToken(FILE(STRING(dataPath + "waypoints/z.dataWaypoints" << problemID)));
    *descFileWaypoints << "feas#time#plan#planID" << endl;
    descFileRRT = new rai::FileToken(FILE(STRING(dataPath + "rrt/z.dataRRT" << problemID)));
    *descFileRRT << "feas#time#plan#planID#actionNum#pathNodes" << endl;
    descFileLGP = new rai::FileToken(FILE(STRING(dataPath + "lgp/z.dataLGP" << problemID)));
    *descFileLGP << "feas#time#plan#planID#" << endl;
  }

  for (uint i = 0; i < 1; i++)
  { // 50 plans
    // Create skeleton node to get action sequence
    auto sket_node = std::dynamic_pointer_cast<rai::LGPComp2_Skeleton>(root.createNewChild(i));
    if(!sket_node) {
      cout << "Failed to create skeleton node " << i << endl;
      continue;
    }
    sket_node->compute(); // find actions sequence
    
    if(!sket_node->isComplete || !sket_node->isFeasible) {
      cout << "Skeleton node " << i << " failed or incomplete" << endl;
      continue;
    }
  
    cout<< "plan number " << i << ", action sequence: " << sket_node->actionSequence << endl;
    
    // Try to find one feasible waypoint solution
    uint num_tries = 0;
    bool found_feasible_waypoint = false;
    std::shared_ptr<rai::LGPComp2_Waypoints> feasible_ways_node = nullptr;
    
    while(!found_feasible_waypoint && num_tries < 500)
    {
      num_tries++;
      
      // Create waypoints node with random seed
      auto ways_node = std::dynamic_pointer_cast<rai::LGPComp2_Waypoints>(
        sket_node->createNewChild(num_tries));
      if(!ways_node) {
        cout << "Failed to create waypoints node" << endl;
        continue;
      }
      
      while(!ways_node->isComplete) {
        ways_node->compute();
      }
      double time_waypoints = ways_node->c;
      
      if(logVerbose){
        *descFileWaypoints << ways_node->isFeasible << "#" << time_waypoints << "#"
                           << "#" << sket_node->actionSequence << "#" << i << endl;
      }
      
      if (ways_node->isFeasible)
      {
        found_feasible_waypoint = true;
        feasible_ways_node = ways_node;
        cout << "Found feasible waypoint solution at try " << num_tries << endl;
      }
      else {
        // Clean up this waypoint node
        ways_node->children.clear();
      }
    }
    
    if(!found_feasible_waypoint) {
      cout << "Could not find feasible waypoint solution for plan " << i << " after " << num_tries << " tries" << endl;
      sket_node->children.clear();
      continue;
    }
    
    // Now we have a feasible waypoint solution
    // Run 100 RRT trials with different seeds
    uint T = feasible_ways_node->komoWaypoints->T;
    cout << "Running " << num_seed_trials << " RRT trials with different seeds..." << endl;
    uint lgp_trial = 0;
    uint rrt_trial = 0;
    while(lgp_trial < num_seed_trials) {
      bool all_rrt_feasible = true;
      std::shared_ptr<rai::LGPComp2_RRTpath> rrt_node = nullptr;
      rai::Array<std::shared_ptr<rai::LGPComp2_RRTpath>> rrt_nodes(T);
      uint total_path_nodes = 0;
      
      for (uint t = 0; t < T; t++)
      {
        // Create RRT node for this segment with seed based on trial number
        if(t == 0) {
          rrt_node = std::dynamic_pointer_cast<rai::LGPComp2_RRTpath>(
            feasible_ways_node->createNewChild(rrt_trial));
        } else {
          rrt_node = std::dynamic_pointer_cast<rai::LGPComp2_RRTpath>(
            rrt_node->createNewChild(rrt_trial));
        }
        
        if(!rrt_node) {
          cout << "Failed to create RRT node for segment " << t << endl;
          all_rrt_feasible = false;
          break;
        }
        
        rrt_nodes(t) = rrt_node;
        
        // Compute RRT until complete
        while(!rrt_node->isComplete) {
          rrt_node->compute();
        }
        double time_rrt = rrt_node->c;
        
        // Count path nodes
        uint path_nodes = 0;
        if(rrt_node->path.N > 0) {
          path_nodes = rrt_node->path.d0;
          total_path_nodes += path_nodes;
        }
        
        if(logVerbose){
          *descFileRRT << rrt_node->isFeasible << "#" << time_rrt << "#" << sket_node->actionSequence << "#" << i << "#" << t 
                       << "#" << path_nodes << endl;
        }
        
        if (!rrt_node->isFeasible)
        {
          all_rrt_feasible = false;
          break;
        }
      }

      auto path_node = std::dynamic_pointer_cast<rai::LGPComp2_OptimizePath>(
        rrt_node->createNewChild(lgp_trial));
      if(!path_node) {
        // Clear and continue
        for(uint t = 0; t < rrt_nodes.N; t++) {
          if(rrt_nodes(t)) rrt_nodes(t)->children.clear();
        }
        rrt_nodes.clear();
        continue;
      }
      
      // Compute path optimization until complete
      while(!path_node->isComplete) {
        path_node->compute();
      }
      double time_path = path_node->c;
      
      if(logVerbose){
        *descFileLGP << path_node->isFeasible << "#" << time_path << "#" << sket_node->actionSequence << "#" << i << endl;
      }
      
      if(rrt_trial % 10 == 0) {
        cout << "  RRT trial " << lgp_trial << "/" << num_seed_trials << endl;
      }
      
      // Clear children of all rrt_nodes after this trial
      for(uint t = 0; t < rrt_nodes.N; t++) {
        if(rrt_nodes(t)) rrt_nodes(t)->children.clear();
      }
      rrt_nodes.clear();
      rrt_trial++;
    }

    
    // Clear waypoint node children
    if(feasible_ways_node) feasible_ways_node->children.clear();
    
    // Clear all child nodes created during trials
    sket_node->children.clear();
    
    cout << "Plan " << i << " completed: seed effect analysis done" << endl;
  }
  
  if(logVerbose) {
    delete descFileWaypoints;
    delete descFileRRT;
    delete descFileLGP;
  }
}

void naiveIteration(rai::LGPComp2_root& root, int num_trials, int problemID, int verbose = 10, bool logVerbose = false)
{
  uint trials = num_trials;
  rai::setParameter<int>("KOMO/verbose", verbose - 2);
  // rai::Array<double> jointState = {};
  rai::FileToken* descFileWaypoints = nullptr; 
  rai::FileToken* descFileRRT = nullptr;
  rai::FileToken* descFileLGP = nullptr;
  if(logVerbose){
    descFileWaypoints = new rai::FileToken(FILE(STRING(dataPath + "waypoints/z.dataWaypoints" << problemID)));
    *descFileWaypoints << "feas#time#evals#plan#planID" << endl;
    descFileRRT = new rai::FileToken(FILE(STRING(dataPath + "rrt/z.dataRRT" << problemID)));
    *descFileRRT << "feas#time#evals#plan#planID#actionNum#q0#qf" << endl;
    descFileLGP = new rai::FileToken(FILE(STRING(dataPath + "lgp/z.dataLGP" << problemID)));
    *descFileLGP << "feas#time#evals#plan#planID#RRTPath" << endl;
  }
  int countFeas = 0;
  int countNotFeas = 0;
  double avgtimeFeas = 0;
  double avgtimeNotFeas = 0;
  rai::Array<double> times_waypoints;
  rai::Array<int> feasibilities_waypoints;
  rai::Array<int> evals_waypoints;

  for (uint i = 0; i < 50; i++)
  { // 50 plans
    // Create skeleton node to get action sequence
    auto sket_node = std::dynamic_pointer_cast<rai::LGPComp2_Skeleton>(root.createNewChild(i));
    if(!sket_node) {
      cout << "Failed to create skeleton node " << i << endl;
      continue;
    }
    sket_node->compute(); // find actions sequence
    
    if(!sket_node->isComplete || !sket_node->isFeasible) {
      cout << "Skeleton node " << i << " failed or incomplete" << endl;
      continue;
    }
  

    cout<< "plan number " << i << ", action sequence: " << sket_node->actionSequence << endl;
    
    uint num_successes = 0;
    uint num_tries = 0;
    while(num_successes < trials && num_tries < 1000)
    {
      num_tries++;
      if(num_tries % 100 == 0) {
        cout << "  Trial " << num_tries << ", successes so far: " << num_successes << endl;
      }
      
      // Create waypoints node with random seed
      auto ways_node = std::dynamic_pointer_cast<rai::LGPComp2_Waypoints>(
        sket_node->createNewChild(num_tries));
      if(!ways_node) {
        cout << "Failed to create waypoints node" << endl;
        continue;
      }
      
      // Compute waypoints until complete
      uint evals = 0;
      while(!ways_node->isComplete) {
        ways_node->compute();
        evals++;
      }
      double time_waypoints = ways_node->c;
      
      times_waypoints.append(time_waypoints);
      evals_waypoints.append(evals);
      
      if (!ways_node->isFeasible)
      {
        countNotFeas++;
        avgtimeNotFeas += (time_waypoints - avgtimeNotFeas) / double(countNotFeas);
        feasibilities_waypoints.append(0);
        continue;
      }
      else{
        countFeas++;
        num_successes++;
        feasibilities_waypoints.append(1);
        avgtimeFeas += (time_waypoints - avgtimeFeas) / double(countFeas);
      }
      
      //-- try all rrt problems
      uint T = ways_node->komoWaypoints->T;
      bool all_rrt_feasible = true;
      std::shared_ptr<rai::LGPComp2_RRTpath> rrt_node = nullptr;
      rai::Array<std::shared_ptr<rai::LGPComp2_RRTpath>> rrt_nodes(T);
      rai::Array<double> concat_path;  // Collect path data immediately
      
      for (uint t = 0; t < T; t++)
      {
        // Create RRT node for this segment
        if(t == 0) {
          rrt_node = std::dynamic_pointer_cast<rai::LGPComp2_RRTpath>(
            ways_node->createNewChild(0));
        } else {
          rrt_node = std::dynamic_pointer_cast<rai::LGPComp2_RRTpath>(
            rrt_node->createNewChild(0));
        }
        
        if(!rrt_node) {
          cout << "Failed to create RRT node for segment " << t << endl;
          all_rrt_feasible = false;
          break;
        }
        
        rrt_nodes(t) = rrt_node;
        
        // Compute RRT until complete
        evals = 0;
        while(!rrt_node->isComplete) {
          rrt_node->compute();
          evals++;
        }
        double time_rrt = rrt_node->c;
        
        // Get q0 and qT for logging - copy before potential invalidation
        arr q0, qT;
        if(logVerbose) {
          q0 = rrt_node->q0;
          qT = rrt_node->qT;
        }
        
        if(logVerbose){
          *descFileRRT << rrt_node->isFeasible << "#" << time_rrt << "#" << evals 
                       << "#" << sket_node->actionSequence << "#" << i << "#" << t 
                       << "#" << q0 << "#" << qT << endl;
        }
        
        if (!rrt_node->isFeasible)
        {
          cout << "rrt fail " << t << endl;
          if (verbose > 1 && rrt_node->rrt) rrt_node->rrt->view(true, STRING("rrt path " << t));
          all_rrt_feasible = false;
          break;
        }
        
        // Collect path data immediately - use path member, not rrt (which gets reset to null)
        if(logVerbose && rrt_node && rrt_node->path.N > 0) {
          try {
            concat_path.append(path_resampleLinear(rrt_node->path, 10));
          } catch(...) {
            cout << "Warning: Failed to resample path for segment " << t << endl;
          }
        }
      }
      
      if (!all_rrt_feasible) {
        // Clear children of all rrt_nodes before continuing
        for(uint t = 0; t < rrt_nodes.N; t++) {
          if(rrt_nodes(t)) rrt_nodes(t)->children.clear();
        }
        continue;
      }
      
      // Create path optimization node
      auto path_node = std::dynamic_pointer_cast<rai::LGPComp2_OptimizePath>(
        rrt_node->createNewChild(0));
      if(!path_node) {
        cout << "Failed to create path optimization node" << endl;
        continue;
      }
      
      // Compute path optimization until complete
      evals = 0;
      while(!path_node->isComplete) {
        path_node->compute();
        evals++;
      }
      double time_path = path_node->c;
      
      if (verbose > 1 && path_node->komoPath)
        path_node->komoPath->view(false, STRING("full path init"));
      
      if(logVerbose){
        *descFileLGP << path_node->isFeasible << "#" << time_path << "#" << evals 
                     << "#" << sket_node->actionSequence << "#" << i << "#" << concat_path << endl;
      }
      
      if (!path_node->isFeasible)
      {
        cout << "lgp fail" << endl;
        if(path_node && path_node->sol.ret) {
          auto ret = path_node->sol.ret;
          cout << ret->done << " " << ret->sos << " " << ret->f << " " << ret->ineq << " " << ret->eq << endl;
        }
      }
      // Clear all rrt_node children
      for(uint t = 0; t < rrt_nodes.N; t++) {
        if(rrt_nodes(t)) rrt_nodes(t)->children.clear();
      }
      rrt_nodes.clear();
      
      // Clear ways_node children
      if(ways_node) ways_node->children.clear();
      
      // Clear collected paths to free memory
      concat_path.clear();
    }
    
    // Clear all child nodes created during trials to free memory and prevent dangling pointers
    sket_node->children.clear();
    
    cout << "Plan " << i << " completed: " << num_successes << "/" << num_tries << " successful trials" << endl;
    if (countFeas == 0) cout << "average time feas ways: nan" << endl;
    else cout << "average time feas ways: " << avgtimeFeas << endl;
    if (countFeas < trials) cout << "average time not feas ways: " << avgtimeNotFeas << endl;
    else cout << "average time not feas ways: nan" << endl;
    avgtimeFeas = 0;
    avgtimeNotFeas = 0;
    countFeas = 0;
    countNotFeas = 0;
    if(logVerbose){
      *descFileWaypoints << feasibilities_waypoints << "#" << times_waypoints << "#" << evals_waypoints 
                         << "#" << sket_node->actionSequence << "#" << i << endl;
    }
    times_waypoints.clear();
    feasibilities_waypoints.clear();
    evals_waypoints.clear();
  }
}

//===========================================================================

int main(int argn, char **argv) {
  rai::initCmdLine(argn, argv);
  rnd.seed(3);
      // rai::Configuration C = getRandomConfiguration(false, 10, true);
      // rai::FileToken ConfFile = FILE(STRING(dataPath + "z.conf10.g"));
      // C.write(ConfFile);
      // C.view(true);
      // return 0;

  int num_problems = 1;   // default
  int agent_id = 2;        // default
  bool logVerbose = true;  // default
  int verbose_level = 0;   // default
  int num_trials = 500;   // default

  for (int i = 1; i < argn; ++i) {
      std::string arg = argv[i];

      if (arg == "--num-problems" && i + 1 < argn) {
          num_problems = std::stoi(argv[++i]);
      }
      else if (arg == "--agent-id" && i + 1 < argn) {
          agent_id = std::stoi(argv[++i]);
      }
      else if (arg == "--log-verbose" && i + 1 < argn) {
          logVerbose = std::stoi(argv[++i]);
      }
      else if (arg == "--verbose" && i + 1 < argn) {
          verbose_level = std::stoi(argv[++i]);
          cout << "Setting verbose level to " << verbose_level << endl;
      }
      else if (arg == "--data-path" && i + 1 < argn) {
          dataPath = argv[++i];
      }
      else if(arg == "--num-trials" && i + 1 < argn) {
          num_trials = std::stoi(argv[++i]);
      }
      else if (arg == "--help") {
          std::cout << "Usage: " << argv[0] << " [--num_problems N] [--agent-id ID] [--log-verbose 0/1] [--verbose LEVEL] [--data-path PATH] [--num-trials N]" << std::endl;
          return 0;
      }
  }
  if(logVerbose){
    dataPath = dataPath+"/data_raw/agent_" + STRING(agent_id) + "/";
    std::filesystem::create_directories((const char*)dataPath);
    std::filesystem::create_directories((const char*)(dataPath + "waypoints/"));
    std::filesystem::create_directories((const char*)(dataPath + "rrt/"));
    std::filesystem::create_directories((const char*)(dataPath + "lgp/"));
    std::filesystem::create_directories((const char*)(dataPath + "configs/"));
  }
  str problem = STRING("2blocks-temp" << agent_id);
  str lgpFile = "temp_lgp_files/" + problem + ".lgp";
  for(int j=0; j<num_problems; j++) {
    cout << "================== Problem " << j << " ==================" << endl;
    // rai::Configuration C = getRandomConfiguration(false, agent_id, true);
    rai::Configuration C(dataPath+"configs/z.conf" << j << ".g");
    C.view(true);
    // if(logVerbose) {
    //   rai::FileToken ConfFile = FILE(STRING(dataPath + "configs/z.conf" << j << ".g"));
    //   C.write(ConfFile);
    //   C.view(true);
    // }
    auto tamp = rai::default_LGP_TAMP_Abstraction(C, lgpFile);
    ifstream file(lgpFile);
    rai::String line;
    uint numObjects = 0;
    while(file.good()) {
      line.read(file, "", "\n", true);
      if(line.startsWith("terminal:")) {
        rai::String terminalStr = line.getSubString(10, -1); // Skip "terminal: "
        for(uint i = 0; i < terminalStr.N; i++) {
          if(terminalStr(i) == '(') numObjects++;
        }
        break;
      }
    }
    file.close();
    
    rai::Graph lgpConfig(lgpFile);

    
    auto root = make_shared<rai::LGPComp2_root>(
        C, *tamp,
        lgpConfig.get<StringA>("lifts", {}),
        lgpConfig.get<str>("terminalSkeleton", {}), 0);
    // naiveIteration(*root, num_trials, j, verbose_level, logVerbose);
       checkSeedEffect(*root, num_trials, j, verbose_level, logVerbose);
  }


  return 0;
}