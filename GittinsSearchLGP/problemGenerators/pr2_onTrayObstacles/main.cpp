#include <LGP/LGP_TAMP_Abstraction.h>
#include <Optim/NLP_Solver.h>
#include <PathAlgos/RRT_PathFinder.h>
#include <Kin/frame.h>
#include <Kin/viewer.h>
#include <iostream>
#include <LGP/LGP_computers2.h>
#include <KOMO/pathTools.h>
#include <Search/ComputeNode.h>
#include <Search/AStar.h>
#include <Core/array.h>
#include <Core/util.h>
#include <Core/graph.h>

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

void solveProblem(const str &configPath, const str &lgpFilePath)
{
  uint evalLimit = rai::getParameter<double>("LGP/evalLimit", 1000.);
  rai::NodeGlobal opt2; 

  cout << "============== Solving pr2_onTrayObstacles with ELS ==============" << endl;
  cout << "Config file: " << configPath << endl;
  cout << "LGP file: " << lgpFilePath << endl;
  cout << "Solver: " << opt2.solver << endl;
   
  // Load configuration and LGP problem
  rai::Configuration C(configPath); 
  // C.view(true);
  // return;
  auto tamp = rai::default_LGP_TAMP_Abstraction(C, lgpFilePath);
  // for(int i=0; i<50; i++) cout << tamp->getNewActionSequence() << endl; // test the LGP_TAMP_Abstraction interface
  // return;
  uint numObjects = countObjectsInLgpFile(lgpFilePath);
  cout << "Number of objects: " << numObjects << endl;

  // Initialize predictor and info
  auto info = std::make_shared<rai::LGP2_GlobalInfo>(); 
  auto predictor = std::make_shared<rai::NodePredictor>(  
      info->predictionType, info->solver, info->modelsDir.p);

  // Load LGP config and create root node
  rai::Graph lgpConfig(lgpFilePath);
  auto root = make_shared<rai::LGPComp2_root>(
      C, *tamp,
      lgpConfig.get<StringA>("lifts", {}),
      lgpConfig.get<str>("terminalSkeleton", {}), 
      0, // seed
      predictor, 
      info);

  // Create A* search
  rai::AStar astar(root);

  // Search for solution 
  uint c_tot = 0;
  bool solFound = false;
   
  while (root->c_tot < evalLimit && astar.queue.N)
  {
    if (c_tot != uint(root->c_tot))
    { 
      c_tot = root->c_tot; 
      cout << "Current c_tot: " << c_tot 
           << ", steps: " << astar.steps 
           << ", mem size: " << astar.mem.N << endl;
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
      solFound = true;
      uint numSkeletons = root->children.N;
      
      cout << "\n============== SOLUTION FOUND! ==============" << endl;
      cout << "Steps: " << astar.steps << endl;
      cout << "c_tot: " << root->c_tot << endl;
      cout << "meta_c_tot: " << root->meta_c_tot << endl;
      cout << "gittins_c_tot: " << root->gittins_c_tot << endl;
      cout << "inference_c_tot: " << root->inference_c_tot << endl;
      cout << "Number of skeletons: " << numSkeletons << endl;
      cout << "Total solutions: " << astar.solutions.N << endl;
      cout << "Feasible solutions: " << solutions << endl;
      
      // Print the compute invested in each node along the found solution path
      auto solNode = astar.solutions.last();
      cout << "\nSolution path compute:" << endl;
      while (solNode)
      {
        auto compNode = static_cast<rai::ComputeNode *>(solNode);
        cout << "  Node: " << *solNode 
             << " | c: " << compNode->c 
             << " | c_tot: " << compNode->c_tot << endl;
        solNode = solNode->parent;
      }
      cout << "=============================================" << endl;
      
      break;
    }
  }

  if (!solFound)
  {
    cout << "\n============== NO SOLUTION FOUND ==============" << endl;
    cout << "Evaluation limit reached: " << evalLimit << endl;
    cout << "Final c_tot: " << root->c_tot << endl;
    cout << "Steps: " << astar.steps << endl;
    cout << "Number of skeletons generated: " << root->children.N << endl;
    cout << "=============================================" << endl;
  }
} 

//===========================================================================
  
int main(int argn, char **argv)
{
  rai::initCmdLine(argn, argv);

  str configPath = "pr2-onTray.g";
  str lgpFilePath = "pr2-onTray.lgp"; 
    // Alternative: Use PROJECT_ROOT environment variable
    // str configPath = "$PROJECT_ROOT/lgp-benchmarks/robot-pnp/pr2-onTray.g";
  // str lgpFilePath = "$PROJECT_ROOT/lgp-benchmarks/robot-pnp/pr2-onTray.lgp";

  solveProblem(configPath, lgpFilePath);

  return 0;
}
