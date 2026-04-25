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

#include <Core/array.h> 
#include <Core/util.h>
#include <Core/graph.h>
#include <math.h>
#include <LGP/LGP_Tool.h> 
#include <LGP/LGP_SkeletonTool.h>
#include <future>

//===========================================================================

void procEval(const char *prefix, uint K)
{
  arrA X(K);
  for (uint k = 0; k < K; k++) 
  {
    FILE(STRING(prefix << k << ".dat")) >> X(k);
  }

  rai::String name = prefix;
  name << "VAR.dat";
  ofstream out(name);

  for (uint t = 0; t < X(0).d0; t++)
  {
    for (uint i = 0; i < X(0).d1; i++)
    {
      double m = 0., v = 0.;
      for (uint k = 0; k < K; k++)
      { 
        double x = X(k)(t, i);
        m += x;
        v += x * x;
      }
      m /= double(K);
      v = ::sqrt(v / double(K) - m * m + 1e-10);
      v /= ::sqrt(double(K));
      out << m << ' ' << v << ' ';
    }
    out << endl;  
  }
}

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
      // cout << "Current c_tot: " << c_tot << ", steps: " << astar.steps << ", mem size: " << astar.mem.N << endl;
    }
    astar.step(false);
    uint solutions = 0;
    for (rai::TreeSearchNode *n : astar.solutions)
    {
      if (n->isFeasible)
        solutions++;
    }
    if (!solFound && solutions > 0)
    { // write first time solution is found
      uint numSkeletons = root->children.N;
      filStop << astar.steps << ',' << root->c_tot << ',' << root->meta_c_tot << ',' << root->gittins_c_tot << ',' << root->inference_c_tot << ',' << numObjects << ','
              << numSkeletons << ',' << 1 << endl;
      solFound = true;
      cout << astar.steps << ' ' << c_tot << ' ' << root->c_tot << ' '
           << astar.mem.N << ' ' << astar.solutions.N << ' ' << solutions << endl;
      // print the compute invested in each node along the found solution path
      auto solNode = astar.solutions.last();
      cout << "Solution found with compute path:" << endl;
      while (solNode)
      {
        auto compNode = static_cast<rai::ComputeNode *>(solNode);
        cout << "Node: " << *solNode << " with c: " << compNode->c << " c_tot: " << compNode->c_tot << endl;
        solNode = solNode->parent;
      }
      return root->c_tot;
    }
  }

  if (!solFound)
  {
    filStop << astar.steps << ',' << root->c_tot << ',' << root->meta_c_tot << ',' << root->gittins_c_tot << ',' << root->inference_c_tot << ',' << numObjects << ',' << root->children.N << ',' << 0 << endl;
  }
  return root->c_tot;
}

//===========================================================================

double ex4_lgpSolver(rai::GittinsSearch &gittinsSearch, rai::ComputeNode *root, uint numObjects, ofstream &filStop)
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
    { // write first time solution is found
      uint numSkeletons = root->children.N;
      filStop << gittinsSearch.steps << ',' << root->c_tot << ',' << root->meta_c_tot << ',' << root->gittins_c_tot << ',' << root->inference_c_tot << ',' << numObjects << ','
              << numSkeletons << ',' << 1 << endl;
      solFound = true;
      cout << gittinsSearch.steps << ' ' << root->c_tot << ' ' << root->c_tot << ' '
           << gittinsSearch.mem.N << ' ' << gittinsSearch.solutions.N << ' ' << solutions << endl;
      // print the compute invested in each node along the found solution path
      auto solNode = gittinsSearch.solutions.last();
      cout << "Solution found with compute path:" << endl;
      while (solNode)
      {
        auto compNode = static_cast<rai::ComputeNode *>(solNode);
        cout << "Node: " << *solNode << " with c: " << compNode->c << " c_tot: " << compNode->c_tot << endl;
        solNode = solNode->getGittinsParent();
      }
      return root->c_tot;
    }
  }

  if (!solFound)
  {  
    filStop << gittinsSearch.steps << ',' << root->c_tot << ',' << root->meta_c_tot << ',' << root->gittins_c_tot << ',' << root->inference_c_tot << ',' << numObjects << ',' << root->children.N << ',' << 0 << endl;
  }
  return root->c_tot;
}

//===========================================================================

struct HyperParams
{
  double level_cP;
  double skeleton_wP;
  double waypoint_w0;
  double skeleton_w0;
  double waypoint_wP;
};

double runGroundTruthExperiment(int numIterations, int numObjects, int totalGoals, int blockedGoals, str configPath, str lgpFilePath, double dataPercentage = 0.2)
{
  rai::NodeGlobal opt2;
  rai::Rnd rnd;
  int seed_shift = opt2.runSeed;
  double ctot_running_average = 0.;
  std::string finalFilePath = "";

  // Initialize NodePredictor once (will be reused across all iterations)
  auto info = std::make_shared<rai::LGP2_GlobalInfo>();
  cout << "predictiontype: " << info->predictionType << endl;
  
  // Append data percentage to modelsDir
  if (info->predictionType == "GNN") {
    std::ostringstream percentStr;
    percentStr << std::fixed << std::setprecision(1) << dataPercentage;
    str modelsDir_with_percent = STRING(info->modelsDir << "p" << percentStr.str() << "/");
    info->modelsDir = modelsDir_with_percent;
    cout << "Using models from: " << info->modelsDir << endl;
  }

  cout << "skeletonWP: " << info->skeleton_wP << endl;
  auto predictor = std::make_shared<rai::NodePredictor>(info->predictionType, info->solver, info->device, info->modelsDir.p);

  // Create output file once for all iterations
  rai::String key = "groundTruthTest_";
  key << opt2.solver << "_";
  if (opt2.solver == "ELS")
    key << opt2.level_cP << "_" << info->skeleton_wP << "_" << info->skeleton_w0 << "_" << info->waypoint_wP << "_" << info->waypoint_w0;
  if (opt2.solver == "GITTINS")
    key << info->predictionType << "_" << info->numWaypoints << "_" << info->numTaskPlans;
  
  // Add data percentage to key
  key << "_p" << std::fixed << std::setprecision(1) << dataPercentage;

  ofstream filStop;
  // Get current timestamp
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);
  auto now_tm = *std::localtime(&now_time_t);
  std::ostringstream timestamp;
  timestamp << std::setfill('0')
            << std::setw(2) << (now_tm.tm_mon + 1) // month
            << std::setw(2) << now_tm.tm_mday      // day
            << "_"
            << std::setw(2) << now_tm.tm_hour // hour
            << std::setw(2) << now_tm.tm_min  // minute
            << std::setw(2) << now_tm.tm_sec; // second
 

    rai::String resultsDir = STRING("results/" << numObjects << "_" << totalGoals << "_" << blockedGoals << "/groundTruthTest/");
    std::filesystem::create_directories((const char *)resultsDir);
    filStop.open(STRING(resultsDir << key << "_" << timestamp.str() << ".STOP.dat"));
    filStop << "steps,ctot,metaCtot,gittinsCtot,inferenceCtot,numObjects,numSkeletons,success" << endl; // ctot is time spend in the compute nodes, metaCtot is time spent overall on metareasoning, gittinsCtot is time spent computing the gittins indices (without the inference), inferenceCtot is time spent in the predictor inference

  for (uint j = 0; j < numIterations; j++) 
  { 
    cout << "=== Iteration " << j << " ===" << endl;
    rai::Configuration C(configPath);
    auto tamp = rai::default_LGP_TAMP_Abstraction(C, lgpFilePath);
    uint numObjects = countObjectsInLgpFile(lgpFilePath);
    rai::Graph lgpConfig(lgpFilePath);
    auto root = make_shared<rai::LGPComp2_root>(
        C, *tamp,
        lgpConfig.get<StringA>("lifts", {}),
        lgpConfig.get<str>("terminalSkeleton", {}), seed_shift + j, predictor, info);
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

    ctot_running_average = (ctot_running_average * j + ctot) / (j + 1);

    cout << "Iteration " << j << " - c_tot: " << ctot
         << " | Running average: " << ctot_running_average << endl;
  }
 
  filStop.close();  

  cout << "\nFinal running average c_tot over " << numIterations << " runs: " << ctot_running_average << endl;

  return ctot_running_average;
}

std::string runExperiment(int numObjects, int totalGoals, int blockedGoals,
                     int numIterations = 10, bool saveResults = true,
                     const std::string &experimentName = "",
                     std::shared_ptr<rai::LGP2_GlobalInfo> info = nullptr,
                     bool randomConfig = true,
                     int numObjLowerBound = 2,
                     int numObjUpperBound = 5,
                     int numGoalsUpperBound = 5,
                     int numBlockedGoalsUpperBound = 3,
                     double dataPercentage = 0.2)
{
  rai::NodeGlobal opt2;
  rai::Rnd rnd;
  int seed_shift = opt2.runSeed;
  double ctot_running_average = 0.;
  std::string finalFilePath = "";

  // Initialize NodePredictor once (will be reused across all iterations)
  if (!info) info = std::make_shared<rai::LGP2_GlobalInfo>();
  
  // Append data percentage to modelsDir 
  if (info->predictionType == "GNN") {
    std::ostringstream percentStr;
    percentStr << std::fixed << std::setprecision(1) << dataPercentage;
    str modelsDir_with_percent = STRING(info->modelsDir << "datasize_" << percentStr.str() << "/");
    info->modelsDir = modelsDir_with_percent;
    cout << "Using models from: " << info->modelsDir << endl;
  }
  
  cout << "skeletonWP: " << info->skeleton_wP << endl;
  cout << "device: " << info->device << endl;
  auto predictor = std::make_shared<rai::NodePredictor>(info->predictionType, info->solver, info->device, info->modelsDir.p);

  // Create output file once for all iterations
  rai::String key = "";
  if (!experimentName.empty()) 
  {
    key << experimentName << "_";
  }
  key << opt2.solver << "_";
  if (opt2.solver == "ELS")
    key << opt2.level_cP << "_" << info->skeleton_wP << "_" << info->skeleton_w0 << "_" << info->waypoint_wP << "_" << info->waypoint_w0;
  if (opt2.solver == "GITTINS")
    key << info->numWaypoints << "_" << info->numTaskPlans;
  
  // Add data percentage to key
  key << "_p" << std::fixed << std::setprecision(1) << dataPercentage;

  ofstream filStop;
  if (saveResults)
  {
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_tm = *std::localtime(&now_time_t);
    std::ostringstream timestamp;
    timestamp << std::setfill('0')
              << std::setw(2) << (now_tm.tm_mon + 1) // month
              << std::setw(2) << now_tm.tm_mday      // day
              << "_"
              << std::setw(2) << now_tm.tm_hour // hour
              << std::setw(2) << now_tm.tm_min  // minute
              << std::setw(2) << now_tm.tm_sec; // second

    // Create results directory based on problem parameters
    if (randomConfig)
    { 
      cout << "creating dir" <<endl;
      rai::String resultsDir = STRING("results/randomConfig_obj" << numObjLowerBound << "_" << numObjUpperBound << "_goals" << numGoalsUpperBound << "_blocked" << numBlockedGoalsUpperBound << "/");
      std::filesystem::create_directories((const char *)resultsDir);
      finalFilePath = STRING(resultsDir << key << "_" << timestamp.str() << ".STOP.dat");
      filStop.open(finalFilePath);
      filStop <<"steps,ctot,metaCtot,gittinsCtot,inferenceCtot,numObjects,numSkeletons,success" << endl;
    }
    else
    {
      rai::String resultsDir = STRING("results/" << numObjects << "_" << totalGoals << "_" << blockedGoals << "/");
      std::filesystem::create_directories((const char *)resultsDir);
      finalFilePath = STRING(resultsDir << key << "_" << timestamp.str() << ".STOP.dat");
      filStop.open(finalFilePath);
      filStop <<"steps,ctot,metaCtot,gittinsCtot,inferenceCtot,numObjects,numSkeletons,success" << endl;
    }
  }

  for (uint j = 0; j < numIterations; j++)
  { 
    cout << "=== Iteration " << j << " ===" << endl;
    if (randomConfig)
    {
      numObjects = rnd.uni_int(numObjLowerBound, numObjUpperBound);
      totalGoals = rnd.uni_int(numObjects, numGoalsUpperBound);
      blockedGoals = rnd.uni_int(0, std::min(totalGoals - 1, numBlockedGoalsUpperBound)); 
    }
    rai::Configuration C = getRandomConfiguration(0, numObjects, totalGoals, blockedGoals);
    // C.view(true);
    str lgpFile = STRING("../../problemGenerators/randomBlocks/temp_lgp_files/randomBlocks_temp" << 0 << ".lgp");
    auto tamp = rai::default_LGP_TAMP_Abstraction(C, lgpFile);
    uint numObjects = countObjectsInLgpFile(lgpFile);
    rai::Graph lgpConfig(lgpFile);
    auto root = make_shared<rai::LGPComp2_root>(
        C, *tamp,
        lgpConfig.get<StringA>("lifts", {}),
        lgpConfig.get<str>("terminalSkeleton", {}), seed_shift + j, predictor, info);
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

    ctot_running_average = (ctot_running_average * j + ctot) / (j + 1);

    cout << "Iteration " << j << " - c_tot: " << ctot
         << " | Running average: " << ctot_running_average << endl;
  }

  if (saveResults)
  {
    filStop.close();
  }

  cout << "\nFinal running average c_tot over " << numIterations << " runs: " << ctot_running_average << endl;

  return finalFilePath;
}

void hyperparameterTuning(int numObjects, int totalGoals, int blockedGoals, bool randomConfig, int numIterations,
                          int numObjLowerBound = 2,
                          int numObjUpperBound = 5,
                          int numGoalsUpperBound = 5,
                          int numBlockedGoalsUpperBound = 3,
                          double dataPercentage = 0.2)
{
  rai::NodeGlobal opt2;
  auto info = std::make_shared<rai::LGP2_GlobalInfo>();

  if (opt2.solver != "ELS")
  {
    cout << "Hyperparameter tuning is only supported for ELS solver" << endl;
    return;
  }

  // Define hyperparameter search space
  std::vector<double> level_cP_values = {0.0, 0.5, 1.0, 2.0, 3.0};
  std::vector<double> wP_shared_values = {1.0, 2.0, 3.0, 4.0}; // shared between skeleton_wP and waypoint_wP
  std::vector<double> waypoint_w0_values = {1.0, 2.0, 5.0, 10.0};
  std::vector<double> skeleton_w0_values = {1.0, 2.0, 5.0};

  double best_avg = std::numeric_limits<double>::max();
  HyperParams best_params;

  cout << "\n========================================" << endl;
  cout << "Starting Hyperparameter Tuning for ELS" << endl;
  cout << "========================================\n"
       << endl;

  // Prepare the aggregation summary file
  rai::String summaryDir = STRING("results/" << numObjects << "_" << totalGoals << "_" << blockedGoals << "/");
  std::filesystem::create_directories((const char *)summaryDir);
  ofstream aggFile(STRING(summaryDir << "tuning_summary_aggregated.csv"));
  aggFile << "level_cP,wP,w0_way,w0_skel,avg_steps,med_steps,avg_ctot,med_ctot,avg_meta,med_meta,avg_gittins,med_gittins,avg_inference,med_inference,avg_skel,avg_succ" << endl;

  int total_combinations = level_cP_values.size() * wP_shared_values.size() *
                           waypoint_w0_values.size() * skeleton_w0_values.size();
  int current_combination = 0;

  for (double level_cP : level_cP_values)
  {
    for (double wP_shared : wP_shared_values)
    { 
      for (double waypoint_w0 : waypoint_w0_values)
      {
        for (double skeleton_w0 : skeleton_w0_values)
        {
          current_combination++;

          cout << "\n--- Configuration " << current_combination << "/" << total_combinations << " ---" << endl;
          cout << "level_cP: " << level_cP << ", wP_shared: " << wP_shared
               << ", waypoint_w0: " << waypoint_w0 << ", skeleton_w0: " << skeleton_w0 << endl;

          // Set hyperparameters (skeleton_wP and waypoint_wP share the same value)
          opt2.set_level_cP(level_cP);
          info->set_skeleton_wP(wP_shared);
          info->set_waypoint_w0(waypoint_w0);
          info->set_skeleton_w0(skeleton_w0);
          info->set_waypoint_wP(wP_shared);
          cout << "Set waypoint_wP to: " << info->waypoint_wP << " (wP_shared=" << wP_shared << ")" << endl;

          // Run experiment
          std::ostringstream expName;
          expName << "tune_" << level_cP << "_" << wP_shared << "_"
                  << waypoint_w0 << "_" << skeleton_w0;

          std::string resultFile = runExperiment(numObjects, totalGoals, blockedGoals, numIterations, true, expName.str(), info, randomConfig, numObjLowerBound, numObjUpperBound, numGoalsUpperBound, numBlockedGoalsUpperBound, dataPercentage);

          // --- Aggregation logic starts here ---
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
            steps.push_back(row[0]); ctot.push_back(row[1]); meta.push_back(row[2]);
            gittins.push_back(row[3]); inference.push_back(row[4]);
            skel_sum += row[6]; succ_sum += row[7];
            count++;
          }

          auto calc_med = [](std::vector<double> v) {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            return v.size() % 2 == 0 ? (v[v.size()/2 - 1] + v[v.size()/2]) / 2 : v[v.size()/2];
          };
          auto calc_avg = [](const std::vector<double>& v) {
            double sum = 0; for(double d : v) sum += d;
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
          // --- Aggregation logic ends here ---

          cout << "Average c_tot: " << current_avg_ctot << endl;

          if (current_avg_ctot < best_avg)
          {
            best_avg = current_avg_ctot;
            best_params.level_cP = level_cP;
            best_params.skeleton_wP = wP_shared;
            best_params.waypoint_w0 = waypoint_w0;
            best_params.skeleton_w0 = skeleton_w0;
            best_params.waypoint_wP = wP_shared;

            cout << "*** NEW BEST! ***" << endl;
          }
        }
      } 
    }
  }

  aggFile.close(); // Close aggregated file

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

  // Save best hyperparameters to file (your original logic)
  rai::String resultsDir = STRING("results/" << numObjects << "_" << totalGoals << "_" << blockedGoals << "/");
  std::filesystem::create_directories((const char *)resultsDir);

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

  ofstream bestFile(STRING(resultsDir << "best_hyperparams_" << timestamp.str() << ".txt"));
  bestFile << "Best average c_tot: " << best_avg << endl;
  bestFile << "level_cP: " << best_params.level_cP << endl;
  bestFile << "skeleton_wP: " << best_params.skeleton_wP << endl;
  bestFile << "waypoint_w0: " << best_params.waypoint_w0 << endl;
  bestFile << "skeleton_w0: " << best_params.skeleton_w0 << endl;
  bestFile << "waypoint_wP: " << best_params.waypoint_wP << endl;
  bestFile.close();
}



int main(int argn, char **argv)
{
  rai::initCmdLine(argn, argv);

  bool tuning = rai::getParameter<bool>("tuning", false);
  int numObjects = rai::getParameter<int>("numObjects", 1);
  int totalGoals = rai::getParameter<int>("totalGoals", 1);
  int blockedGoals = rai::getParameter<int>("blockedGoals", 1);
  bool randomConfig = rai::getParameter<bool>("randomConfig", true);
  int numIterations = rai::getParameter<int>("numIterations", 10);
  
  // Parameters for bounds (read from rai.cfg)
  int numObjLowerBound = rai::getParameter<int>("numObjLowerBound", 2);
  int numObjUpperBound = rai::getParameter<int>("numObjUpperBound", 5);
  int numGoalsUpperBound = rai::getParameter<int>("numGoalsUpperBound", 5);
  int numBlockedGoalsUpperBound = rai::getParameter<int>("numBlockedGoalsUpperBound", 3);
  bool saveResults = rai::getParameter<bool>("saveResults", true);
  bool percentageExperiment = rai::getParameter<bool>("percentageExperiment", false);
  
  // Data percentage parameter for model selection
  double dataPercentage = rai::getParameter<double>("dataPercentage", 0.2);

  rai::NodeGlobal opt2;
  rnd.seed(opt2.runSeed);

  if (rai::getParameter<bool>("groundTruthTest", false))
  {
    runGroundTruthExperiment(numIterations, numObjects, totalGoals, blockedGoals, rai::getParameter<str>("configPath"), rai::getParameter<str>("lgpFilePath"), dataPercentage);
  }

  else if (tuning)
  {
    hyperparameterTuning(numObjects, totalGoals, blockedGoals, randomConfig, numIterations, numObjLowerBound, numObjUpperBound, numGoalsUpperBound, numBlockedGoalsUpperBound, dataPercentage);
  }
  else if (percentageExperiment){
    // Define the three configurations
    struct Config { 
        int lowObj; 
        int highObj; 
        int highGoals; 
        int highBlocked; 
    };

    std::vector<Config> configs = {
      {2, 2, 2, 2},
      {4, 4, 4, 1},
      {3, 3, 3, 2},
    }; 

    cout << "\n>>> Starting Percentage Experiment (Sequential) <<<" << endl;

    for (size_t i = 0; i < configs.size(); ++i) {
      cout << "\n--- Running Configuration " << i + 1 << "/" << configs.size() << " ---" << endl;
      cout << "Bounds: Objects[" << configs[i].lowObj << "," << configs[i].highObj 
           << "], Max Goals: " << configs[i].highGoals 
           << "], Max Blocked: " << configs[i].highBlocked << endl;

      // We pass 'true' for randomConfig to ensure the bounds are respected inside runExperiment
      runExperiment(
        numObjects, 
        totalGoals, 
        blockedGoals, 
        numIterations, 
        saveResults, 
        "", 
        nullptr, 
        true, // randomConfig
        configs[i].lowObj, 
        configs[i].highObj, 
        configs[i].highGoals, 
        configs[i].highBlocked, 
        dataPercentage
      );
    }
    
    cout << "\n>>> Percentage Experiment Complete <<<" << endl;
  }
  else
  {
    runExperiment(numObjects, totalGoals, blockedGoals, numIterations, saveResults, "", nullptr, randomConfig, numObjLowerBound, numObjUpperBound, numGoalsUpperBound, numBlockedGoalsUpperBound, dataPercentage);
  }
 
  return 0;
}