#include <LGP/LGP_TAMP_Abstraction.h>
#include <Optim/NLP_Solver.h>
#include <PathAlgos/RRT_PathFinder.h>
#include <Kin/frame.h>
#include <Kin/viewer.h>
#include "../../problemGenerators/randomBlocks/ConfigurationGenerator.h"
#include <filesystem>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <LGP/LGP_computers2.h>
#include <KOMO/pathTools.h>
#include <Search/ComputeNode.h>
#include <Search/AStar.h>
#include <LGP/GittinsSearch.h>
#include <Search/BanditProcess.h>

#include <Core/array.h>  
#include <Core/util.h>
#include <Core/graph.h>
#include <math.h>
#include <LGP/LGP_Tool.h> 
#include <LGP/LGP_SkeletonTool.h>
#include <future>

//===========================================================================
// Utility Functions
//===========================================================================

uint countObjectsInLgpFile(const str &lgpFile)
{
  ifstream file(lgpFile);
  rai::String line;
  uint numObjects = 0;
  while (file.good())
  {
    line.read(file, "", "\n", true);
    if (line.startsWith("terminal:"))
    {
      rai::String terminalStr = line.getSubString(10, -1); // Skip "terminal: "
      for (uint i = 0; i < terminalStr.N; i++)
      {
        if (terminalStr(i) == '(')
          numObjects++;
      }
      break;
    }
  }
  file.close(); 
  return numObjects;
}

std::string getConfigDirectory(int numObjLowerBound, int numObjUpperBound, 
                                int numGoalsUpperBound, int numBlockedGoalsUpperBound)
{
  return (char*)STRING("dataSizeExperiment/obj" << numObjLowerBound << "_" << numObjUpperBound 
                << "_goals" << numGoalsUpperBound << "_blocked" << numBlockedGoalsUpperBound << "/");
}

std::string getResultsDirectory(int numObjLowerBound, int numObjUpperBound, 
                                 int numGoalsUpperBound, int numBlockedGoalsUpperBound)
{
  return (char*)STRING(getConfigDirectory(numObjLowerBound, numObjUpperBound, 
                                   numGoalsUpperBound, numBlockedGoalsUpperBound) << "results/");
}

//===========================================================================
// MODE 1: Configuration Generation
//===========================================================================

void createConfigs(int numObjLowerBound, int numObjUpperBound,
                   int numGoalsUpperBound, int numBlockedGoalsUpperBound,
                   int numIterations, int seed)
{
  cout << "\n╔════════════════════════════════════════════════════╗" << endl;
  cout << "║   GENERATING CONFIGURATIONS                        ║" << endl;
  cout << "╚════════════════════════════════════════════════════╝\n" << endl;
  
  cout << "Config Family: Objects[" << numObjLowerBound << "," << numObjUpperBound 
       << "], Goals[" << numGoalsUpperBound << "], Blocked[" << numBlockedGoalsUpperBound << "]" << endl;
  cout << "Num Iterations: " << numIterations << endl;
  cout << "Seed: " << seed << endl;

  // Set up directories
  std::string baseDir = getConfigDirectory(numObjLowerBound, numObjUpperBound, 
                                           numGoalsUpperBound, numBlockedGoalsUpperBound);
  std::string configDir = (char*)STRING(baseDir << "configs/");
  std::filesystem::create_directories(configDir);
  
  // Seed the random number generator
  rnd.seed(seed);
  
  // Save metadata
  ofstream metaFile(STRING(configDir << "metadata.txt"));
  metaFile << "numObjLowerBound: " << numObjLowerBound << endl;
  metaFile << "numObjUpperBound: " << numObjUpperBound << endl;
  metaFile << "numGoalsUpperBound: " << numGoalsUpperBound << endl;
  metaFile << "numBlockedGoalsUpperBound: " << numBlockedGoalsUpperBound << endl;
  metaFile << "numIterations: " << numIterations << endl;
  metaFile << "seed: " << seed << endl;
  metaFile.close();
  
  // Save config parameters for each iteration
  ofstream configListFile(STRING(configDir << "config_list.csv"));
  configListFile << "iteration,numObjects,totalGoals,blockedGoals,configFile,lgpFile" << endl;
  
  cout << "\nGenerating configurations..." << endl;
  for(int i = 0; i < numIterations; i++)
  {
    // Sample random parameters within bounds
    int numObjects = rnd.uni_int(numObjLowerBound, numObjUpperBound);
    int totalGoals = rnd.uni_int(numObjects, numGoalsUpperBound);
    int blockedGoals = rnd.uni_int(0, std::min(totalGoals, numBlockedGoalsUpperBound)); 
    
    if (i % 10 == 0) {
      cout << "  Config " << i << "/" << numIterations << ": " 
           << numObjects << " obj, " << totalGoals << " goals, " 
           << blockedGoals << " blocked" << endl;
    }
    
    // Generate configuration
    rai::Configuration C = getRandomConfiguration(i, numObjects, totalGoals, blockedGoals);
    
    // Save configuration file
    str configFile = STRING(configDir << "config_" << i << ".g");
    rai::FileToken ConfFile = FILE(configFile);
    C.write(ConfFile);
    
    // Copy LGP file to config directory
    str tempLgpFile = STRING("../../problemGenerators/randomBlocks/temp_lgp_files/randomBlocks_temp" << i << ".lgp");
    str savedLgpFile = STRING(configDir << "lgp_" << i << ".lgp");
    std::filesystem::copy_file((char*)tempLgpFile, (char*)savedLgpFile, std::filesystem::copy_options::overwrite_existing);
    
    // Record in config list
    configListFile << i << "," << numObjects << "," << totalGoals << "," 
                   << blockedGoals << "," << configFile << "," << savedLgpFile << endl;
  }
  
  configListFile.close();
  
  cout << "\n✓ Successfully generated " << numIterations << " configurations" << endl;
  cout << "✓ Saved to: " << configDir << "\n" << endl;
}

//===========================================================================
// Solver Functions
//===========================================================================

double ex4_lgpSolver(rai::AStar &astar, rai::ComputeNode *root, uint numObjects, ofstream &filStop)
{
  uint evalLimit = rai::getParameter<double>("LGP/evalLimit");
  rai::NodeGlobal opt2;

  rai::String key = "randomBlocks";
  key << '_' << opt2.solver;
  if (opt2.solver == "ELS")
    key << opt2.level_wP << opt2.level_cP;

  LOG(0) << "============== run: " << key;

  uint c_tot = 0;
  bool solFound = false;
  while (root->c_tot < evalLimit && astar.queue.N)
  {
    if (c_tot != uint(root->c_tot))
    {
      c_tot = root->c_tot;
    }
    astar.step(false);
    uint solutions = 0;
    for (rai::TreeSearchNode *n : astar.solutions)
    {
      if (n->isFeasible)
        solutions++;
    }
    if (!solFound && solutions > 0)
    {
      uint numSkeletons = root->children.N;
      std::string skeletonName = "";
      auto solNode = astar.solutions.last();
      cout << "Solution found with compute path:" << endl;
      while (solNode)
      {
        auto compNode = static_cast<rai::ComputeNode *>(solNode);
        cout << "Node: " << *solNode << " with c: " << compNode->c 
             << " c_tot: " << compNode->c_tot << endl;
        if (skeletonName.empty() && std::string(compNode->name.p).find("Skeleton") != std::string::npos)
          skeletonName = compNode->name.p;
        solNode = solNode->parent;
      }
      filStop << astar.steps << ',' << root->c_tot << ',' << root->meta_c_tot << ',' 
              << root->gittins_c_tot << ',' << root->inference_c_tot << ',' << numObjects << ','
              << numSkeletons << ',' << 1 << ',' << skeletonName << endl;
      solFound = true;
      cout << astar.steps << ' ' << c_tot << ' ' << root->c_tot << ' '
           << astar.mem.N << ' ' << astar.solutions.N << ' ' << solutions << endl;
      return root->c_tot;
    }
  }

  if (!solFound)
  {
    filStop << astar.steps << ',' << root->c_tot << ',' << root->meta_c_tot << ',' 
            << root->gittins_c_tot << ',' << root->inference_c_tot << ',' << numObjects << ',' 
            << root->children.N << ',' << 0 << ',' << "" << endl;
  }
  return root->c_tot;
}

double ex4_lgpSolver(rai::GittinsSearch &gittinsSearch, rai::ComputeNode *root, 
                     uint numObjects, ofstream &filStop)
{
  uint evalLimit = rai::getParameter<double>("LGP/evalLimit");
  rai::NodeGlobal opt2;

  rai::String key = "randomBlocks";
  key << '_' << opt2.solver;
  if (opt2.solver == "ELS")
    key << opt2.level_wP << opt2.level_cP;

  LOG(0) << "============== run: " << key;

  bool solFound = false;
  while (root->c_tot < evalLimit)
  {
    gittinsSearch.step();
    if (!gittinsSearch.queue.N)
      break;
    uint solutions = 0;
    for (GittinsNode *n : gittinsSearch.solutions)
    {
      if (n->isFeasible)
        solutions++;
    }
    if (!solFound && solutions > 0)
    {
      uint numSkeletons = root->children.N;
      std::string skeletonName = "";
      auto solNode = gittinsSearch.solutions.last();
      cout << "Solution found with compute path:" << endl;
      while (solNode)
      {
        auto compNode = static_cast<rai::ComputeNode *>(solNode);
        cout << "Node: " << *solNode << " with c: " << compNode->c 
             << " c_tot: " << compNode->c_tot << endl;
        if (skeletonName.empty() && std::string(compNode->name.p).find("Skeleton") != std::string::npos)
          skeletonName = compNode->name.p;
        solNode = solNode->getGittinsParent();
      }
      filStop << gittinsSearch.steps << ',' << root->c_tot << ',' << root->meta_c_tot << ',' 
              << root->gittins_c_tot << ',' << root->inference_c_tot << ',' << numObjects << ','
              << numSkeletons << ',' << 1 << ',' << skeletonName << endl;
      solFound = true;
      cout << gittinsSearch.steps << ' ' << root->c_tot << ' ' << root->c_tot << ' '
           << gittinsSearch.mem.N << ' ' << gittinsSearch.solutions.N << ' ' << solutions << endl;
      return root->c_tot;
    }
  }

  if (!solFound)
  {  
    filStop << gittinsSearch.steps << ',' << root->c_tot << ',' << root->meta_c_tot << ',' 
            << root->gittins_c_tot << ',' << root->inference_c_tot << ',' << numObjects << ',' 
            << root->children.N << ',' << 0 << ',' << "" << endl;
  }
  return root->c_tot;
}

//===========================================================================
// Warm up run for running with Gittins
//===========================================================================
std::shared_ptr<rai::NodePredictor> warmUpGittins(
    double dataPercentage, int modelSeed, bool useDatasizeSubdir)
{
  cout << "\n╔════════════════════════════════════════════════════╗" << endl;
  cout << "║   WARMING UP GITTINS SOLVER                      ║" << endl;
  cout << "╚════════════════════════════════════════════════════╝\n" << endl;

  auto info = std::make_shared<rai::LGP2_GlobalInfo>();

  // Set up model directory (mirrors runExperimentOnSavedConfigs)
  if (info->predictionType == "GNN") {
    if (useDatasizeSubdir) {
      std::ostringstream percentStr;
      percentStr << std::fixed << std::setprecision(1) << dataPercentage;
      str modelsDir_with_percent = STRING(info->modelsDir << "datasize_" << percentStr.str()
                                          << "/seed_" << modelSeed << "/");
      info->modelsDir = modelsDir_with_percent;
    }
    cout << "Using models from: " << info->modelsDir << endl;
  }

  auto predictor = std::make_shared<rai::NodePredictor>(info->predictionType, info->solver,
                                                         info->device, info->modelsDir.p);

  str configDir = STRING(getConfigDirectory(2, 2, 2, 2)<< "configs/");

  str configFile = STRING(configDir << "config_0.g");
  str lgpFile    = STRING(configDir << "lgp_0.lgp");

  if (!std::filesystem::exists((char*)configFile) || !std::filesystem::exists((char*)lgpFile))
  {
    cerr << "WARNING: Cannot warm up – missing config_0 files at " << configDir << endl;
    cerr << "Returning predictor without warm-up run." << endl;
    return predictor;
  }

  cout << "Loading config 0 for warm-up run..." << endl;
  rai::Configuration C(configFile);
  auto tamp = rai::default_LGP_TAMP_Abstraction(C, lgpFile);
  uint numObjects = countObjectsInLgpFile(lgpFile);
  rai::Graph lgpConfig(lgpFile);

  auto root = make_shared<rai::LGPComp2_root>(
      C, *tamp,
      lgpConfig.get<StringA>("lifts", {}),
      lgpConfig.get<str>("terminalSkeleton", {}),
      0, predictor, info);

  // Discard results – we only care about warming up the predictor
  ofstream devnull("/dev/null");
  devnull << "steps,ctot,metaCtot,gittinsCtot,inferenceCtot,numObjects,numSkeletons,success,skeletonName" << endl;

  rai::GittinsSearch gittinsSearch(root);
  ex4_lgpSolver(gittinsSearch, root.get(), numObjects, devnull);

  cout << "✓ Warm-up complete!\n" << endl;
  return predictor;
}

//===========================================================================
// MODE 2: Run Experiment on Saved Configs (GITTINS or ELS)
//===========================================================================

void runExperimentOnSavedConfigs(int numObjLowerBound, int numObjUpperBound,
                                  int numGoalsUpperBound, int numBlockedGoalsUpperBound,
                                  int numIterations, double dataPercentage, 
                                  int seed, int modelSeed, const std::string& experimentName = "",
                                  bool useDatasizeSubdir = false,
                                  std::shared_ptr<rai::NodePredictor> warmPredictor = nullptr)
{
  cout << "\n╔════════════════════════════════════════════════════╗" << endl;
  cout << "║   RUNNING EXPERIMENT ON SAVED CONFIGS              ║" << endl;
  cout << "╚════════════════════════════════════════════════════╝\n" << endl;
  
  rai::NodeGlobal opt2;
  auto info = std::make_shared<rai::LGP2_GlobalInfo>();
  
  // Print experiment info
  cout << "Config Family: Objects[" << numObjLowerBound << "," << numObjUpperBound 
       << "], Goals[" << numGoalsUpperBound << "], Blocked[" << numBlockedGoalsUpperBound << "]" << endl;
  cout << "Solver: " << opt2.solver << endl;
  cout << "Data Percentage: " << dataPercentage << endl;
  cout << "Model Seed: " << modelSeed << endl;
  cout << "Num Iterations: " << numIterations << endl;
  
  // Set up directories
  str configDir = STRING(getConfigDirectory(numObjLowerBound, numObjUpperBound,
                                                     numGoalsUpperBound, numBlockedGoalsUpperBound) 
                                 << "configs/");
  std::string resultsDir = getResultsDirectory(numObjLowerBound, numObjUpperBound,
                                               numGoalsUpperBound, numBlockedGoalsUpperBound);
  if (useDatasizeSubdir && opt2.solver == "GITTINS") {
    std::ostringstream percentStr;
    percentStr << std::fixed << std::setprecision(1) << dataPercentage;
    resultsDir += "datasize_" + percentStr.str() + "/";
  }
  std::filesystem::create_directories(resultsDir);
  
  // Check if configs exist
  if (!std::filesystem::exists((char*)STRING(configDir << "config_list.csv")))
  {
    cerr << "ERROR: Configurations not found at " << configDir << endl;
    cerr << "Please run with mode=generate first!" << endl;
    return; 
  }
  
  // Set up model directory for GNN
  if (info->predictionType == "GNN") {
    if (useDatasizeSubdir) {
      std::ostringstream percentStr;
      percentStr << std::fixed << std::setprecision(1) << dataPercentage;
      str modelsDir_with_percent = STRING(info->modelsDir << "datasize_" << percentStr.str() << "/seed_" << modelSeed << "/");
      info->modelsDir = modelsDir_with_percent;
    }
    cout << "Using models from: " << info->modelsDir << endl; 
  }
   
  // Initialize predictor (reuse warmed-up predictor if provided)
  cout << "Initializing predictor..." << endl;
  auto predictor = warmPredictor
      ? warmPredictor
      : std::make_shared<rai::NodePredictor>(info->predictionType, info->solver,
                                              info->device, info->modelsDir.p);
  
  // Build output filename
  rai::String key = "";
  if (!experimentName.empty()) { 
    key << experimentName << "_";
  }
  key << opt2.solver << "_";
  if (opt2.solver == "ELS") {  
    key << opt2.level_cP << "_" << info->skeleton_wP << "_" << info->skeleton_w0 
        << "_" << info->waypoint_wP << "_" << info->waypoint_w0;
  }
  if (opt2.solver == "GITTINS") {
    key << info->numWaypoints << "_" << info->numTaskPlans;
  }
  key << "_p" << std::fixed << std::setprecision(1) << dataPercentage;
  if (opt2.solver == "GITTINS") { 
    key << "_ms" << modelSeed;
  }
  
  // Add timestamp
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);
  auto now_tm = *std::localtime(&now_time_t);
  std::ostringstream timestamp;
  timestamp << std::setfill('0')
            << std::setw(2) << (now_tm.tm_mon + 1)
            << std::setw(2) << now_tm.tm_mday << "_"
            << std::setw(2) << now_tm.tm_hour
            << std::setw(2) << now_tm.tm_min
            << std::setw(2) << now_tm.tm_sec;
  
  std::string outputFile = (char*)STRING(resultsDir << key << "_" << timestamp.str() << ".STOP.dat");
  ofstream filStop(outputFile);
  filStop << "steps,ctot,metaCtot,gittinsCtot,inferenceCtot,numObjects,numSkeletons,success,skeletonName" << endl;
  
  cout << "\nResults will be saved to: " << outputFile << "\n" << endl;
  
  // Run experiments
  int seed_shift = seed;
  double ctot_running_average = 0.;
  
  for (int i = 0; i < numIterations; i++)
  {
    cout << "=== Iteration " << i << "/" << numIterations << " ===" << endl;
    
    // Load configuration
    str configFile = STRING(configDir << "config_" << i << ".g");
    str lgpFile = STRING(configDir << "lgp_" << i << ".lgp");
    
    if (!std::filesystem::exists((char*)configFile) || !std::filesystem::exists((char*)lgpFile))
    {
      cerr << "ERROR: Missing config files for iteration " << i << endl;
      continue;
    }
    
    cout << "Loading: " << configFile << endl;
    rai::Configuration C(configFile);
    
    auto tamp = rai::default_LGP_TAMP_Abstraction(C, lgpFile);
    uint numObjects = countObjectsInLgpFile(lgpFile);
    rai::Graph lgpConfig(lgpFile);
    
    auto root = make_shared<rai::LGPComp2_root>(
        C, *tamp,
        lgpConfig.get<StringA>("lifts", {}),
        lgpConfig.get<str>("terminalSkeleton", {}), 
        seed_shift + i, predictor, info);
     
    double ctot;
    if (opt2.solver == "GITTINS" && info->predictionType == "GNN")
    {
      rai::GittinsSearch gittinsSearch(root);
      ctot = ex4_lgpSolver(gittinsSearch, root.get(), numObjects, filStop);
    }
    else
    {
      rai::AStar astar(root);
      ctot = ex4_lgpSolver(astar, root.get(), numObjects, filStop);
    }
    
    ctot_running_average = (ctot_running_average * i + ctot) / (i + 1);
    cout << "Iteration " << i << " - c_tot: " << ctot
         << " | Running average: " << ctot_running_average << endl;
  }
   
  filStop.close();
  
  cout << "\n✓ Experiment complete!" << endl; 
  cout << "✓ Final running average c_tot: " << ctot_running_average << endl;
  cout << "✓ Results saved to: " << outputFile << "\n" << endl;
}  

//===========================================================================
// MODE 3: Hyperparameter Tuning on Saved Configs
//===========================================================================

struct HyperParams
{ 
  double level_cP;
  double skeleton_wP;
  double waypoint_w0;
  double skeleton_w0;
  double waypoint_wP;
};

void runTuningOnSavedConfigs(int numObjLowerBound, int numObjUpperBound,
                              int numGoalsUpperBound, int numBlockedGoalsUpperBound,
                              int numIterations, double dataPercentage, int seed, int modelSeed)
{
  cout << "\n╔════════════════════════════════════════════════════╗" << endl;
  cout << "║   RUNNING HYPERPARAMETER TUNING                    ║" << endl;
  cout << "╚════════════════════════════════════════════════════╝\n" << endl;
  
  rai::NodeGlobal opt2;
  
  if (opt2.solver != "ELS")
  {
    cerr << "ERROR: Tuning only works with ELS solver!" << endl;
    cerr << "Set solver: ELS in your config file." << endl;
    return;
  }
  
  auto info = std::make_shared<rai::LGP2_GlobalInfo>();
  
  cout << "Config Family: Objects[" << numObjLowerBound << "," << numObjUpperBound 
       << "], Goals[" << numGoalsUpperBound << "], Blocked[" << numBlockedGoalsUpperBound << "]" << endl;
  cout << "Data Percentage: " << dataPercentage << endl;
  cout << "Num Iterations per config: " << numIterations << endl;
  
  // Set up directories
  str configDir = STRING(getConfigDirectory(numObjLowerBound, numObjUpperBound,
                                                     numGoalsUpperBound, numBlockedGoalsUpperBound) 
                                 << "configs/");
  str resultsDir = STRING(getResultsDirectory(numObjLowerBound, numObjUpperBound,
                                                       numGoalsUpperBound, numBlockedGoalsUpperBound) 
                                  << "tuning/");
  std::filesystem::create_directories((char*)resultsDir);
  
  // Define hyperparameter search space
  std::vector<double> level_cP_values = {0.5, 1.0, 2.0, 3.0};
  std::vector<double> wP_shared_values = {1.0, 2.0, 3.0};
  std::vector<double> waypoint_w0_values = {1.0, 10.0};
  std::vector<double> skeleton_w0_values = {1.0};
  
  double best_avg = std::numeric_limits<double>::max();
  HyperParams best_params;
  
  // Create aggregation file
  ofstream aggFile(STRING(resultsDir << "tuning_summary_aggregated.csv"));
  aggFile << "level_cP,wP,w0_way,w0_skel,avg_steps,med_steps,avg_ctot,med_ctot,"
          << "avg_meta,med_meta,avg_gittins,med_gittins,avg_inference,med_inference,"
          << "avg_skel,avg_succ" << endl;
  
  int total_combinations = level_cP_values.size() * wP_shared_values.size() *
                           waypoint_w0_values.size() * skeleton_w0_values.size();
  int current_combination = 0;
  
  cout << "\nTesting " << total_combinations << " hyperparameter combinations...\n" << endl;
  
  for (double level_cP : level_cP_values)
  {
    for (double wP_shared : wP_shared_values)
    { 
      for (double waypoint_w0 : waypoint_w0_values)
      {
        for (double skeleton_w0 : skeleton_w0_values)
        {
          if(wP_shared < level_cP){
            continue;
          }
          current_combination++;
          
          cout << "\n--- Combination " << current_combination << "/" << total_combinations << " ---" << endl;
          cout << "level_cP: " << level_cP << ", wP: " << wP_shared
               << ", w0_way: " << waypoint_w0 << ", w0_skel: " << skeleton_w0 << endl;
          
          // Set hyperparameters
          opt2.set_level_cP(level_cP);
          info->set_skeleton_wP(wP_shared);
          info->set_waypoint_w0(waypoint_w0);
          info->set_skeleton_w0(skeleton_w0);
          info->set_waypoint_wP(wP_shared);
          
          // Build experiment name
          std::ostringstream expName;
          expName << "tune_" << level_cP << "_" << wP_shared << "_"
                  << waypoint_w0 << "_" << skeleton_w0;
          
          // Run experiment with these hyperparameters
          // (We'll call a modified version that doesn't print the banner)
          
          // Set up model directory
          if (info->predictionType == "GNN") {
            std::ostringstream percentStr;
            percentStr << std::fixed << std::setprecision(1) << dataPercentage;
            str modelsDir_with_percent = STRING(info->modelsDir << "p" << percentStr.str() << "/seed_" << modelSeed << "/");
            info->modelsDir = modelsDir_with_percent;
          } 
          
          auto predictor = std::make_shared<rai::NodePredictor>(info->predictionType, 
                                                                 info->solver, 
                                                                 info->device, 
                                                                 info->modelsDir.p);
          
          // Build output filename
          rai::String key = STRING(expName.str() << "_ELS_" 
                                   << opt2.level_cP << "_" << info->skeleton_wP << "_" 
                                   << info->skeleton_w0 << "_" << info->waypoint_wP << "_" 
                                   << info->waypoint_w0 << "_p" << std::fixed 
                                   << std::setprecision(1) << dataPercentage);
          
          auto now = std::chrono::system_clock::now();
          auto now_time_t = std::chrono::system_clock::to_time_t(now);
          auto now_tm = *std::localtime(&now_time_t);
          std::ostringstream timestamp;
          timestamp << std::setfill('0')
                    << std::setw(2) << (now_tm.tm_mon + 1)
                    << std::setw(2) << now_tm.tm_mday << "_"
                    << std::setw(2) << now_tm.tm_hour
                    << std::setw(2) << now_tm.tm_min
                    << std::setw(2) << now_tm.tm_sec;
          
          std::string resultFile = (char*)STRING(resultsDir << key << "_" << timestamp.str() << ".STOP.dat");
          ofstream filStop(resultFile);
          filStop << "steps,ctot,metaCtot,gittinsCtot,inferenceCtot,numObjects,numSkeletons,success,skeletonName" << endl;
          
          // Run on all configs
          int seed_shift = seed;
          for (int i = 0; i < numIterations; i++)
          {
            str configFile = STRING(configDir << "config_" << i << ".g");
            str lgpFile = STRING(configDir << "lgp_" << i << ".lgp");
            
            rai::Configuration C(configFile);
            auto tamp = rai::default_LGP_TAMP_Abstraction(C, lgpFile);
            uint numObjects = countObjectsInLgpFile(lgpFile);
            rai::Graph lgpConfig(lgpFile);
            
            auto root = make_shared<rai::LGPComp2_root>(
                C, *tamp,
                lgpConfig.get<StringA>("lifts", {}),
                lgpConfig.get<str>("terminalSkeleton", {}), 
                seed_shift + i, predictor, info);
            
            rai::AStar astar(root);
            ex4_lgpSolver(astar, root.get(), numObjects, filStop);
          }
          
          filStop.close();
          
          // Aggregate results
          ifstream fin(resultFile);
          std::string line, header;
          std::getline(fin, header);
          
          std::vector<double> steps, ctot, meta, gittins, inference;
          double skel_sum = 0, succ_sum = 0;
          int count = 0;
          
          while (std::getline(fin, line)) {
            std::stringstream ss(line);
            std::string val;
            std::vector<double> row;
            while (std::getline(ss, val, ',')) row.push_back(std::stod(val));
            if(row.size() < 8) continue;
            steps.push_back(row[0]); 
            ctot.push_back(row[1]); 
            meta.push_back(row[2]);
            gittins.push_back(row[3]); 
            inference.push_back(row[4]);
            skel_sum += row[6]; 
            succ_sum += row[7];
            count++;
          }
          
          auto calc_med = [](std::vector<double> v) {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            return v.size() % 2 == 0 ? (v[v.size()/2 - 1] + v[v.size()/2]) / 2 : v[v.size()/2];
          };
          auto calc_avg = [](const std::vector<double>& v) {
            double sum = 0; 
            for(double d : v) sum += d;
            return v.empty() ? 0.0 : sum / v.size();
          };
          
          double current_avg_ctot = calc_avg(ctot);
          aggFile << level_cP << "," << wP_shared << "," << waypoint_w0 << "," << skeleton_w0 << ","
                  << calc_avg(steps) << "," << calc_med(steps) << ","
                  << current_avg_ctot << "," << calc_med(ctot) << ","
                  << calc_avg(meta) << "," << calc_med(meta) << ","
                  << calc_avg(gittins) << "," << calc_med(gittins) << ","
                  << calc_avg(inference) << "," << calc_med(inference) << ","
                  << (skel_sum/count) << "," << (succ_sum/count) << endl;
          
          cout << "  Average c_tot: " << current_avg_ctot << endl;
          
          if (current_avg_ctot < best_avg)
          {
            best_avg = current_avg_ctot;
            best_params.level_cP = level_cP;
            best_params.skeleton_wP = wP_shared;
            best_params.waypoint_w0 = waypoint_w0;
            best_params.skeleton_w0 = skeleton_w0;
            best_params.waypoint_wP = wP_shared;
            cout << "  *** NEW BEST! ***" << endl;
          }
        }
      } 
    }
  }
  
  aggFile.close();
  
  cout << "\n========================================" << endl;
  cout << "Hyperparameter Tuning Complete" << endl;
  cout << "========================================" << endl; 
  cout << "Best average c_tot: " << best_avg << endl;
  cout << "Best hyperparameters:" << endl;
  cout << "  level_cP: " << best_params.level_cP << endl;
  cout << "  skeleton_wP: " << best_params.skeleton_wP << endl;
  cout << "  waypoint_w0: " << best_params.waypoint_w0 << endl;
  cout << "  skeleton_w0: " << best_params.skeleton_w0 << endl;
  cout << "  waypoint_wP: " << best_params.waypoint_wP << endl;
  
  // Save best parameters
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);
  auto now_tm = *std::localtime(&now_time_t);
  std::ostringstream timestamp;
  timestamp << std::setfill('0')
            << std::setw(2) << (now_tm.tm_mon + 1)
            << std::setw(2) << now_tm.tm_mday << "_"
            << std::setw(2) << now_tm.tm_hour
            << std::setw(2) << now_tm.tm_min
            << std::setw(2) << now_tm.tm_sec;
  
  ofstream bestFile(STRING(resultsDir << "../best_hyperparams_" << timestamp.str() << ".txt"));
  bestFile << "Best average c_tot: " << best_avg << endl;
  bestFile << "level_cP: " << best_params.level_cP << endl;
  bestFile << "skeleton_wP: " << best_params.skeleton_wP << endl;
  bestFile << "waypoint_w0: " << best_params.waypoint_w0 << endl;
  bestFile << "skeleton_w0: " << best_params.skeleton_w0 << endl;
  bestFile << "waypoint_wP: " << best_params.waypoint_wP << endl;
  bestFile.close();
  
  cout << "\n✓ Results saved to: " << resultsDir << "\n" << endl;
}

//===========================================================================
// MODE 4: Hyperparameter Tuning for Gittins Solver
//===========================================================================

struct GittinsHyperParams
{
  int    numWaypoints;
  int    numTaskPlans;
  double beta;
};

void runGittinsTuningOnSavedConfigs(int numObjLowerBound, int numObjUpperBound,
                                     int numGoalsUpperBound, int numBlockedGoalsUpperBound,
                                     int numIterations, int seed,
                                     std::shared_ptr<rai::NodePredictor> warmPredictor = nullptr)
{
  cout << "\n╔════════════════════════════════════════════════════╗" << endl;
  cout << "║   RUNNING GITTINS HYPERPARAMETER TUNING            ║" << endl;
  cout << "╚════════════════════════════════════════════════════╝\n" << endl;

  rai::NodeGlobal opt2;

  if (opt2.solver != "GITTINS")
  {
    cerr << "ERROR: tuneGittins only works with GITTINS solver!" << endl;
    cerr << "Set solver: GITTINS in your config file." << endl;
    return;
  }

  cout << "Config Family: Objects[" << numObjLowerBound << "," << numObjUpperBound
       << "], Goals[" << numGoalsUpperBound << "], Blocked[" << numBlockedGoalsUpperBound << "]" << endl;
  cout << "Num Iterations per config: " << numIterations << endl;

  // Set up directories
  str configDir = STRING(getConfigDirectory(numObjLowerBound, numObjUpperBound,
                                             numGoalsUpperBound, numBlockedGoalsUpperBound)
                         << "configs/");

  // Derive a short name from modelsDir so results are grouped per model set
  std::string _modelsDirName;
  {
    rai::LGP2_GlobalInfo _tmp;
    std::filesystem::path modelsPath = std::string(_tmp.modelsDir.p);
    // strip trailing slash then take the last path component
    if (modelsPath.filename().empty()) modelsPath = modelsPath.parent_path();
    _modelsDirName = modelsPath.filename().string();
  }

  str resultsDir = STRING(getResultsDirectory(numObjLowerBound, numObjUpperBound,
                                               numGoalsUpperBound, numBlockedGoalsUpperBound)
                          << _modelsDirName << "/tuning_gittins/");
  std::filesystem::create_directories((char*)resultsDir);

  // Hyperparameter search space
  std::vector<int>    numWaypoints_values  = {30, 75, 150};   
  std::vector<int>    numTaskPlans_values  = {3, 5, 10};
  std::vector<double> beta_values          = {0.99999}; 

  double best_avg = std::numeric_limits<double>::max();
  GittinsHyperParams best_params;

  ofstream aggFile(STRING(resultsDir << "tuning_gittins_summary.csv"));
  aggFile << "numWaypoints,numTaskPlans,beta,avg_steps,med_steps,avg_ctot,med_ctot,"
          << "avg_meta,med_meta,avg_gittins,med_gittins,avg_inference,med_inference,"
          << "avg_skel,avg_succ" << endl;

  int total_combinations = (int)(numWaypoints_values.size() * numTaskPlans_values.size() * beta_values.size());
  int current_combination = 0;

  cout << "\nTesting " << total_combinations << " hyperparameter combinations...\n" << endl;

  for (int numWaypoints : numWaypoints_values)
  { 
    for (int numTaskPlans : numTaskPlans_values)
    {
      for (double beta : beta_values)
      {
        current_combination++;
        cout << "\n--- Combination " << current_combination << "/" << total_combinations << " ---" << endl;
        cout << "numWaypoints: " << numWaypoints
             << ", numTaskPlans: " << numTaskPlans
             << ", beta: " << beta << endl;

        // Override hyperparameters
        // beta must be set in the global parameter store so that BanditProcess
        // instances created during search pick it up on construction.
        rai::setParameter<double>("Bandit/beta", beta);

        auto info = std::make_shared<rai::LGP2_GlobalInfo>();
        info->set_numWaypoints(numWaypoints);
        info->set_numTaskPlans(numTaskPlans);

        auto predictor = warmPredictor
            ? warmPredictor
            : std::make_shared<rai::NodePredictor>(info->predictionType,
                                                    info->solver,
                                                    info->device,
                                                    info->modelsDir.p);

        // Build output filename
        rai::String key = STRING("GITTINS_nw" << numWaypoints
                                 << "_ntp" << numTaskPlans
                                 << "_b" << beta);

        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        auto now_tm = *std::localtime(&now_time_t);
        std::ostringstream timestamp;
        timestamp << std::setfill('0')
                  << std::setw(2) << (now_tm.tm_mon + 1)
                  << std::setw(2) << now_tm.tm_mday << "_"
                  << std::setw(2) << now_tm.tm_hour
                  << std::setw(2) << now_tm.tm_min
                  << std::setw(2) << now_tm.tm_sec;

        std::string resultFile = (char*)STRING(resultsDir << key << "_" << timestamp.str() << ".STOP.dat");
        ofstream filStop(resultFile);
        filStop << "steps,ctot,metaCtot,gittinsCtot,inferenceCtot,numObjects,numSkeletons,success,skeletonName" << endl;

        // Run on all configs
        int seed_shift = seed;
        for (int i = 0; i < numIterations; i++)
        {
          str configFile = STRING(configDir << "config_" << i << ".g");
          str lgpFile    = STRING(configDir << "lgp_"    << i << ".lgp");

          if (!std::filesystem::exists((char*)configFile) || !std::filesystem::exists((char*)lgpFile))
          {
            cerr << "ERROR: Missing config files for iteration " << i << endl;
            continue;
          }

          rai::Configuration C(configFile);
          auto tamp = rai::default_LGP_TAMP_Abstraction(C, lgpFile);
          uint numObjects = countObjectsInLgpFile(lgpFile);
          rai::Graph lgpConfig(lgpFile);

          auto root = make_shared<rai::LGPComp2_root>(
              C, *tamp,
              lgpConfig.get<StringA>("lifts", {}),
              lgpConfig.get<str>("terminalSkeleton", {}),
              seed_shift + i, predictor, info);

          rai::GittinsSearch gittinsSearch(root);
          ex4_lgpSolver(gittinsSearch, root.get(), numObjects, filStop);
        }

        filStop.close();

        // Aggregate results
        ifstream fin(resultFile);
        std::string line, header;
        std::getline(fin, header);

        std::vector<double> steps, ctot, meta, gittins, inference;
        double skel_sum = 0, succ_sum = 0;
        int count = 0;

        while (std::getline(fin, line)) {
          std::stringstream ss(line);
          std::string val;
          std::vector<double> row;
          while (std::getline(ss, val, ',')) row.push_back(std::stod(val));
          if (row.size() < 8) continue;
          steps.push_back(row[0]);
          ctot.push_back(row[1]);
          meta.push_back(row[2]);
          gittins.push_back(row[3]);
          inference.push_back(row[4]);
          skel_sum += row[6];
          succ_sum += row[7];
          count++;
        }

        auto calc_med = [](std::vector<double> v) {
          if (v.empty()) return 0.0;
          std::sort(v.begin(), v.end());
          return v.size() % 2 == 0 ? (v[v.size()/2 - 1] + v[v.size()/2]) / 2 : v[v.size()/2];
        };
        auto calc_avg = [](const std::vector<double>& v) {
          double sum = 0;
          for (double d : v) sum += d;
          return v.empty() ? 0.0 : sum / v.size();
        };

        double current_avg_ctot = calc_avg(ctot);
        aggFile << numWaypoints << "," << numTaskPlans << "," << beta << ","
                << calc_avg(steps) << "," << calc_med(steps) << ","
                << current_avg_ctot << "," << calc_med(ctot) << ","
                << calc_avg(meta)  << "," << calc_med(meta)  << ","
                << calc_avg(gittins) << "," << calc_med(gittins) << ","
                << calc_avg(inference) << "," << calc_med(inference) << ","
                << (count > 0 ? skel_sum/count : 0.0) << ","
                << (count > 0 ? succ_sum/count : 0.0) << endl;

        cout << "  Average c_tot: " << current_avg_ctot << endl;

        if (current_avg_ctot < best_avg)
        {
          best_avg = current_avg_ctot;
          best_params.numWaypoints = numWaypoints;
          best_params.numTaskPlans = numTaskPlans;
          best_params.beta         = beta;
          cout << "  *** NEW BEST! ***" << endl;
        }
      }
    }
  }

  aggFile.close();

  cout << "\n========================================" << endl;
  cout << "Gittins Hyperparameter Tuning Complete" << endl;
  cout << "========================================" << endl;
  cout << "Best average c_tot: " << best_avg << endl;
  cout << "Best hyperparameters:" << endl;
  cout << "  numWaypoints : " << best_params.numWaypoints << endl;
  cout << "  numTaskPlans : " << best_params.numTaskPlans  << endl;
  cout << "  beta         : " << best_params.beta          << endl;

  // Save best parameters
  auto now2 = std::chrono::system_clock::now();
  auto now_time_t2 = std::chrono::system_clock::to_time_t(now2);
  auto now_tm2 = *std::localtime(&now_time_t2);
  std::ostringstream ts2;
  ts2 << std::setfill('0')
      << std::setw(2) << (now_tm2.tm_mon + 1)
      << std::setw(2) << now_tm2.tm_mday << "_"
      << std::setw(2) << now_tm2.tm_hour
      << std::setw(2) << now_tm2.tm_min
      << std::setw(2) << now_tm2.tm_sec;

  ofstream bestFile(STRING(resultsDir << "../best_gittins_hyperparams_" << ts2.str() << ".txt"));
  bestFile << "Best average c_tot: " << best_avg << endl;
  bestFile << "numWaypoints: "  << best_params.numWaypoints << endl;
  bestFile << "numTaskPlans: "  << best_params.numTaskPlans  << endl;
  bestFile << "beta: "          << best_params.beta          << endl;
  bestFile.close();

  cout << "\n✓ Results saved to: " << resultsDir << "\n" << endl;
}

//===========================================================================
// Main Function
//===========================================================================

int main(int argn, char **argv)
{
  rai::initCmdLine(argn, argv);
  
  // Read mode parameter
  str mode = rai::getParameter<str>("mode", "run");
  
  // Read config family parameters
  int numObjLowerBound = rai::getParameter<int>("numObjLowerBound", 2);
  int numObjUpperBound = rai::getParameter<int>("numObjUpperBound", 5);
  int numGoalsUpperBound = rai::getParameter<int>("numGoalsUpperBound", 5);
  int numBlockedGoalsUpperBound = rai::getParameter<int>("numBlockedGoalsUpperBound", 3);
  
  // Read experiment parameters
  int numIterations = rai::getParameter<int>("numIterations", 100);
  double dataPercentage = rai::getParameter<double>("dataPercentage", 0.2);
  int seed = rai::getParameter<int>("runSeed", 0);
  int modelSeed = rai::getParameter<int>("modelSeed", 42);
  bool useDatasizeSubdir = rai::getParameter<bool>("useDatasizeSubdir", false);
  str solver = rai::getParameter<str>("solver", "GITTINS");
  rai::NodeGlobal opt2;
  opt2.solver = solver;
  
  // Initialize random number generator
  rnd.seed(seed);
  
  cout << "\n╔════════════════════════════════════════════════════╗" << endl;
  cout << "║   DATA SIZE EXPERIMENT                             ║" << endl;
  cout << "╚════════════════════════════════════════════════════╝\n" << endl;
  
  cout << "Mode: " << mode << endl;

  // Warm up the Gittins predictor once before any experiment mode
  std::shared_ptr<rai::NodePredictor> warmedPredictor = nullptr;
  if (solver == "GITTINS" && mode != "generate")
  {
    warmedPredictor = warmUpGittins(dataPercentage, modelSeed, useDatasizeSubdir);
  }

  if (mode == "generate")
  {
    // MODE 1: Generate configurations
    createConfigs(numObjLowerBound, numObjUpperBound, numGoalsUpperBound, 
                  numBlockedGoalsUpperBound, numIterations, seed);
  }
  else if (mode == "run") 
  {
    // MODE 2: Run experiment on saved configs
    str experimentName = rai::getParameter<str>("experimentName", "");
    runExperimentOnSavedConfigs(numObjLowerBound, numObjUpperBound, numGoalsUpperBound,
                                 numBlockedGoalsUpperBound, numIterations, dataPercentage, 
                                 seed, modelSeed, experimentName.p, useDatasizeSubdir,
                                 warmedPredictor);
  }
  else if (mode == "tune")
  {
    // MODE 3: Run hyperparameter tuning
    runTuningOnSavedConfigs(numObjLowerBound, numObjUpperBound, numGoalsUpperBound,
                             numBlockedGoalsUpperBound, numIterations, dataPercentage, seed, modelSeed);
  }
  else if (mode == "tuneGittins")
  {
    // MODE 4: Run Gittins hyperparameter tuning
    runGittinsTuningOnSavedConfigs(numObjLowerBound, numObjUpperBound, numGoalsUpperBound,
                                    numBlockedGoalsUpperBound, numIterations, seed,
                                    warmedPredictor);
  }
  else
  {
    cerr << "\nERROR: Unknown mode '" << mode << "'" << endl;
    cerr << "Valid modes are: generate, run, tune, tuneGittins" << endl;
    return 1;
  }
  
  return 0;
}
