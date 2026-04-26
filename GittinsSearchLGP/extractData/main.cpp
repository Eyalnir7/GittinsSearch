#include <LGP/LGP_TAMP_Abstraction.h>
#include <Kin/frame.h>
#include <Kin/viewer.h>
#include "../problemGenerators/randomBlocks/ConfigurationGenerator.h"
#include "dataCollection.h"
#include <filesystem>
#include <iostream>

rai::Rnd rnd;

//===========================================================================

struct ProgramConfig {
  RAI_PARAM("exp/", int, num_problems, 50)
  RAI_PARAM("exp/", int, agent_id, 0)
  RAI_PARAM("exp/", int, verbose_level, 0)
  RAI_PARAM("exp/", int, num_seed_trials, 3)
  RAI_PARAM("exp/", int, num_plans, 3)
  RAI_PARAM("exp/", int, num_waypoints_tries, 500)
  RAI_PARAM("exp/", int, max_plans, 50)
  RAI_PARAM("exp/", int, start_config_id, 0)
  RAI_PARAM("exp/", int, numObjLowerBound, 2)
  RAI_PARAM("exp/", int, numObjUpperBound, 5)
  RAI_PARAM("exp/", int, numGoalsUpperBound, 5)
  RAI_PARAM("exp/", int, numBlockedGoalsUpperBound, 3)
  RAI_PARAM("exp/", bool, single_config, false)
  RAI_PARAM("exp/", rai::String, dataPath, "../data/randomBlocks1/")
  RAI_PARAM("exp/", rai::String, configPath, "../problemGenerators/pr2_onTrayObstacles/pr2-onTray.g")
  RAI_PARAM("exp/", rai::String, lgpFilesPath, "../problemGenerators/pr2_onTrayObstacles/pr2-onTray.lgp")
};

void parseCommandLineArguments(int argn, char **argv, ProgramConfig& config) {
  for (int i = 1; i < argn; ++i) {
    std::string arg = argv[i];

    if (arg == "--num_problems" && i + 1 < argn) {
      config.num_problems = std::stoi(argv[++i]);
      cout << "[DEBUG] Set num_problems to " << config.num_problems << endl;
    }
    else if (arg == "--agent_id" && i + 1 < argn) {
      config.agent_id = std::stoi(argv[++i]);
      cout << "[DEBUG] Set agent_id to " << config.agent_id << endl;
    }
    else if (arg == "--verbose_level" && i + 1 < argn) {
      config.verbose_level = std::stoi(argv[++i]);
      cout << "[DEBUG] Set verbose_level to " << config.verbose_level << endl;
    }
    else if (arg == "--num_seed_trials" && i + 1 < argn) {
      config.num_seed_trials = std::stoi(argv[++i]);
      cout << "[DEBUG] Set num_seed_trials to " << config.num_seed_trials << endl;
    }
    else if (arg == "--num_plans" && i + 1 < argn) {
      config.num_plans = std::stoi(argv[++i]);
      cout << "[DEBUG] Set num_plans to " << config.num_plans << endl;
    }
    else if (arg == "--num_waypoints_tries" && i + 1 < argn) {
      config.num_waypoints_tries = std::stoi(argv[++i]);
      cout << "[DEBUG] Set num_waypoints_tries to " << config.num_waypoints_tries << endl;
    }
    else if (arg == "--max_plans" && i + 1 < argn) {
      config.max_plans = std::stoi(argv[++i]);
      cout << "[DEBUG] Set max_plans to " << config.max_plans << endl;
    }
    else if (arg == "--start_config_id" && i + 1 < argn) {
      config.start_config_id = std::stoi(argv[++i]);
      cout << "[DEBUG] Set start_config_id to " << config.start_config_id << endl;
    }
    else if (arg == "--numObjLowerBound" && i + 1 < argn) {
      config.numObjLowerBound = std::stoi(argv[++i]);
      cout << "[DEBUG] Set numObjLowerBound to " << config.numObjLowerBound << endl;
    }
    else if (arg == "--numObjUpperBound" && i + 1 < argn) {
      config.numObjUpperBound = std::stoi(argv[++i]);
      cout << "[DEBUG] Set numObjUpperBound to " << config.numObjUpperBound << endl;
    }
    else if (arg == "--numGoalsUpperBound" && i + 1 < argn) {
      config.numGoalsUpperBound = std::stoi(argv[++i]);
      cout << "[DEBUG] Set numGoalsUpperBound to " << config.numGoalsUpperBound << endl;
    }
    else if (arg == "--numBlockedGoalsUpperBound" && i + 1 < argn) {
      config.numBlockedGoalsUpperBound = std::stoi(argv[++i]);
      cout << "[DEBUG] Set numBlockedGoalsUpperBound to " << config.numBlockedGoalsUpperBound << endl;
    }
    else if (arg == "--single_config" && i + 1 < argn) {
      std::string val = argv[++i];
      config.single_config = (val == "true" || val == "1");
      cout << "[DEBUG] Set single_config to " << config.single_config << endl;
    }
    else if (arg == "--dataPath" && i + 1 < argn) {
      config.dataPath = argv[++i];
      cout << "[DEBUG] Set dataPath to " << config.dataPath << endl;
    }
    else if (arg == "--configPath" && i + 1 < argn) {
      config.configPath = argv[++i];
      cout << "[DEBUG] Set configPath to " << config.configPath << endl;
    }
    else if (arg == "--lgpFilesPath" && i + 1 < argn) {
      config.lgpFilesPath = argv[++i];
      cout << "[DEBUG] Set lgpFilesPath to " << config.lgpFilesPath << endl;
    }
    else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --num_problems N                  Number of problems to generate (default: 50)\n"
                << "  --agent_id ID                     Agent ID (default: 0)\n"
                << "  --verbose_level LEVEL             Verbose level (default: 0)\n"
                << "  --dataPath PATH                   Data path\n"
                << "  --num_seed_trials N               Number of seed trials (default: 3)\n"
                << "  --num_plans N                     Number of plans (default: 3)\n"
                << "  --num_waypoints_tries N           Number of waypoints tries (default: 500)\n"
                << "  --max_plans N                     Maximum number of plans (default: 50)\n"
                << "  --start_config_id N               Starting config ID (default: 0)\n"
                << "  --numObjLowerBound N              Lower bound for number of objects (default: 2)\n"
                << "  --numObjUpperBound N              Upper bound for number of objects (default: 5)\n"
                << "  --numGoalsUpperBound N            Upper bound for number of goals (default: 5)\n"
                << "  --numBlockedGoalsUpperBound N     Upper bound for blocked goals (default: 3)\n"
                << "  --single_config BOOL              Use single config mode (default: false)\n"
                << "  --configPath PATH                 Configuration path\n"
                << "  --lgpFilesPath PATH               LGP files path\n"
                << "  --help                            Show this help message\n";
      exit(0);
    }
  }
}
// void parseCommandLineArguments(int argn, char **argv, ProgramConfig& config) {
//   for (int i = 1; i < argn; ++i) {
//       std::string arg = argv[i];

//       if (arg == "--num-problems" && i + 1 < argn) {
//           config.num_problems = std::stoi(argv[++i]);
//       }
//       else if (arg == "--agent-id" && i + 1 < argn) {
//           config.agent_id = std::stoi(argv[++i]);
//       }
//       else if (arg == "--verbose" && i + 1 < argn) {
//           config.verbose_level = std::stoi(argv[++i]);
//           cout << "Setting verbose level to " << config.verbose_level << endl;
//       }
//       else if (arg == "--data-path" && i + 1 < argn) {
//           config.dataPath = argv[++i];
//       }
//       else if(arg == "--num-seed-trials" && i + 1 < argn) {
//           config.num_seed_trials = std::stoi(argv[++i]);
//       }
//       else if(arg == "--num-plans" && i + 1 < argn) {
//           config.num_plans = std::stoi(argv[++i]);
//       }
//       else if(arg == "--num-waypoints-tries" && i + 1 < argn) {
//           config.num_waypoints_tries = std::stoi(argv[++i]);
//       }
//       else if(arg == "--max-plans" && i + 1 < argn) {
//           config.max_plans = std::stoi(argv[++i]);
//       }
//       else if(arg == "--start-config-id" && i + 1 < argn) {
//           config.start_config_id = std::stoi(argv[++i]);
//           cout << "Starting from config ID " << config.start_config_id << endl;
//       }
//       else if(arg == "--num-obj-lower-bound" && i + 1 < argn) {
//           config.numObjLowerBound = std::stoi(argv[++i]);
//       }
//       else if(arg == "--num-obj-upper-bound" && i + 1 < argn) {
//           config.numObjUpperBound = std::stoi(argv[++i]);
//       }
//       else if(arg == "--num-goals-upper-bound" && i + 1 < argn) {
//           config.numGoalsUpperBound = std::stoi(argv[++i]);
//       }
//       else if(arg == "--num-blocked-goals-upper-bound" && i + 1 < argn) {
//           config.numBlockedGoalsUpperBound = std::stoi(argv[++i]);
//       }
//       else if(arg == "--config-path" && i + 1 < argn) {
//           config.configPath = argv[++i];
//       }
//       else if(arg == "--lgp-files-path" && i + 1 < argn) {
//           config.lgpFilesPath = argv[++i];
//       }
//       else if(arg == "--single-config") {
//           config.single_config = true;
//       }
//       else if (arg == "--help") {
//           std::cout << "Usage: " << argv[0] << " [options]\n"
//                     << "Options:\n"
//                     << "  --num-problems N              Number of problems to generate (default: 50)\n"
//                     << "  --agent-id ID                 Agent ID (default: 0)\n"
//                     << "  --verbose LEVEL               Verbose level (default: 0)\n"
//                     << "  --data-path PATH              Data path (default: current)\n"
//                     << "  --num-seed-trials N           Number of seed trials (default: 3)\n"
//                     << "  --num-plans N                 Number of plans (default: 3)\n"
//                     << "  --num-waypoints-tries N       Number of waypoints tries (default: 500)\n"
//                     << "  --max-plans N                 Maximum number of plans to explore (default: 50)\n"
//                     << "  --start-config-id N           Starting config ID (default: 0)\n"
//                     << "  --num-obj-lower-bound N       Lower bound for number of objects (default: 2)\n"
//                     << "  --num-obj-upper-bound N       Upper bound for number of objects (default: 5)\n"
//                     << "  --num-goals-upper-bound N     Upper bound for number of goals (default: 5)\n"
//                     << "  --single-config               Use single configuration from config-path\n"
//                     << "  --num-blocked-goals-upper-bound N  Upper bound for blocked goals (default: 3)\n"
//                     << "  --config-path PATH            Configuration path\n"
//                     << "  --lgp-files-path PATH         LGP files path\n"
//                     << "  --help                        Show this help message\n";
//           exit(0);
//       }
//   }
// }

void randomConfigCollectData(const ProgramConfig& config) {
  for(int j=config.start_config_id; j<config.num_problems; j++) {
    cout << "[DEBUG] ================== Problem " << j << " ==================" << endl;
    cout << "[DEBUG] Generating random configuration" << endl;
    // generate random number between configured bounds
    int numObjects = rnd.uni_int(config.numObjLowerBound, config.numObjUpperBound);
    int numGoals = rnd.uni_int(numObjects, config.numGoalsUpperBound);
    int numBlockedGoals = rnd.uni_int(0, std::min(numGoals, config.numBlockedGoalsUpperBound));
    rai::Configuration C = getRandomConfiguration(config.agent_id, numObjects, numGoals, numBlockedGoals, false);
    // C.view(true);
    cout << "[DEBUG] Random configuration generated" << endl;
    str lgpFile = STRING("../problemGenerators/randomBlocks/temp_lgp_files/randomBlocks_temp" << config.agent_id << ".lgp");
    rai::FileToken ConfFile = FILE(STRING(config.dataPath + "configs/z.conf" << j << ".g"));
    C.write(ConfFile);
    cout << "[DEBUG] Creating TAMP abstraction" << endl;
    auto tamp = rai::default_LGP_TAMP_Abstraction(C, lgpFile);
    cout << "[DEBUG] TAMP abstraction created, reading LGP file" << endl;
    
    rai::Graph lgpConfig(lgpFile);

    
    cout << "[DEBUG] Creating root node" << endl;
    auto root = make_shared<rai::LGPComp2_root>(
        C, *tamp,
        lgpConfig.get<StringA>("lifts", {}),
        lgpConfig.get<str>("terminalSkeleton", {}), config.agent_id*100+j);
    // for(int k=0; k<100; k++){
    //   cout<< root->tamp.getNewActionSequence() << endl;
    // }
    // return 0;
    cout << "[DEBUG] Root node created, calling collectData" << endl;
    collectData(*root, config.num_seed_trials, j, config.num_plans, config.verbose_level, config.dataPath, rnd, false, config.num_waypoints_tries, config.max_plans);
    cout << "[DEBUG] collectData returned for problem " << j << endl;
      //  checkSeedEffect(*root, num_trials, j, verbose_level, logVerbose);
  }
}

void singleConfigCollectData(const ProgramConfig& config) {
  cout << "[DEBUG] ================== Single Config Mode ==================" << endl;
  cout << "[DEBUG] Loading configuration from: " << config.configPath << endl;
  
  rai::Configuration C(config.configPath);
  cout << "[DEBUG] Configuration loaded" << endl;
  
  str lgpFile = STRING(config.lgpFilesPath);
  cout << "[DEBUG] Using LGP file: " << lgpFile << endl;
  
  cout << "[DEBUG] Creating TAMP abstraction" << endl;
  auto tamp = rai::default_LGP_TAMP_Abstraction(C, lgpFile);
  cout << "[DEBUG] TAMP abstraction created" << endl;
  
  rai::Graph lgpConfig(lgpFile);
  
  cout << "[DEBUG] Creating root node" << endl;
  auto root = make_shared<rai::LGPComp2_root>(
      C, *tamp,
      lgpConfig.get<StringA>("lifts", {}),
      lgpConfig.get<str>("terminalSkeleton", {}), config.agent_id);
  
  cout << "[DEBUG] Root node created, calling collectData" << endl;
  collectData(*root, config.num_seed_trials, 0, config.num_plans, config.verbose_level, config.dataPath, rnd, true, config.num_waypoints_tries, config.max_plans);
  cout << "[DEBUG] collectData returned for single config" << endl;
}

int main(int argn, char **argv) {

  ProgramConfig config;
  cout << "[DEBUG] Starting main, initializing command line" << endl;
  rai::initCmdLine(argn, argv, true);
  rnd.seed(3);
      // rai::Configuration C = getRandomConfiguration(false, 10, true);
      // rai::FileToken ConfFile = FILE(STRING(dataPath + "z.conf10.g"));
      // C.write(ConfFile);
      // C.view(true); 
      // return 0;
 
  parseCommandLineArguments(argn, argv, config);
  
  cout << "[DEBUG] Raw parameters read from command line:" << endl;
  cout << "[DEBUG]   agent_id: " << config.agent_id << endl;
  cout << "[DEBUG]   dataPath: " << config.dataPath << endl;
  cout << "[DEBUG]   numObjLowerBound: " << config.numObjLowerBound << endl;
  cout << "[DEBUG]   numObjUpperBound: " << config.numObjUpperBound << endl;
  cout << "[DEBUG]   numGoalsUpperBound: " << config.numGoalsUpperBound << endl;
  cout << "[DEBUG]   numBlockedGoalsUpperBound: " << config.numBlockedGoalsUpperBound << endl;
  cout << "[DEBUG]   num_plans: " << config.num_plans << endl;
  cout << "[DEBUG]   num_seed_trials: " << config.num_seed_trials << endl;
  cout << "[DEBUG]   num_problems: " << config.num_problems << endl;
  cout << "[DEBUG]   num_waypoints_tries: " << config.num_waypoints_tries << endl;
  
  config.dataPath = config.dataPath+"/data_raw/agent_" + STRING(config.agent_id) + "/";
  cout << "[DEBUG] Data path set to: " << config.dataPath << endl;
  std::filesystem::create_directories((const char*)config.dataPath);
  std::filesystem::create_directories((const char*)(config.dataPath + "waypoints/"));
  std::filesystem::create_directories((const char*)(config.dataPath + "rrt/"));
  std::filesystem::create_directories((const char*)(config.dataPath + "lgp/"));
  std::filesystem::create_directories((const char*)(config.dataPath + "configs/"));
  // return 0;
  if (config.single_config) {
    singleConfigCollectData(config); 
  } else { 
    randomConfigCollectData(config);
  } 

  return 0; 
}