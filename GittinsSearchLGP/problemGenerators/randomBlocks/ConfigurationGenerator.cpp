#include "ConfigurationGenerator.h"
#include <Kin/frame.h>
#include <Kin/kin.h>
#include <Core/graph.h>
#include <Core/util.h>
#include <string>
#include <libgen.h>
#include <Core/graph.h>
#include <filesystem>
#include <cstdlib>

// Helper function to get the directory of this source file
std::string getSourceDirectory() {
  // Use PROJECT_ROOT environment variable to construct the absolute path
  // PROJECT_ROOT should point to the root of the project
  const char* projectRoot = std::getenv("PROJECT_ROOT");
  
  if (projectRoot == nullptr) {
    std::cerr << "[ERROR] PROJECT_ROOT environment variable is not set" << std::endl;
    throw std::runtime_error("PROJECT_ROOT environment variable is not set");
  }
  
  std::cerr << "[DEBUG] PROJECT_ROOT = " << projectRoot << std::endl;
  
  std::filesystem::path sourceDir = 
      std::filesystem::path(projectRoot) / "GittinsSearchLGP" / "problemGenerators" / "randomBlocks";
  
  std::cerr << "[DEBUG] Looking for sourceDir at: " << sourceDir.string() << std::endl;
  
  if (!std::filesystem::exists(sourceDir)) {
    std::cerr << "[ERROR] Directory does not exist: " << sourceDir.string() << std::endl;
    throw std::runtime_error("problemGenerators/randomBlocks directory not found at: " + sourceDir.string());
  }
  
  return sourceDir.string();
}   
 
rai::Configuration getRandomConfiguration(int id, int numObjects, int numGoals, int numBlockedGoals, bool movableWalls)
{
  rnd.seed_random(); 
  std::string sourceDir = getSourceDirectory();
  str problem = "base_config"; 

  // Use absolute paths to avoid directory change issues
  std::filesystem::path absLgpFile = std::filesystem::path(sourceDir) / (std::string(problem.p) + ".lgp");
  std::filesystem::path absConfFile = std::filesystem::path(sourceDir) / (std::string(problem.p) + ".g");
  
  str lgpFile = STRING(absLgpFile.string());
  str confFile = STRING(absConfFile.string());
  
  cout << "Using lgp file: " << lgpFile << endl;
  cout << "Using config file: " << confFile << endl;
  rai::Configuration C;
  C.addFile(confFile);
 
  // Add objects
  for(int i = 0; i < numObjects; i++) { 
    char idx = 'A' + i;
    addRandomObject(C, idx, movableWalls);
  }

  // Add goals - first add regular goals (with walls), then blocked goals
  int numRegularGoals = numGoals - numBlockedGoals;
  
  // Add regular goals (with walls)
  for(int i = 0; i < numRegularGoals; i++) {
    char idx = 'A' + i;
    addRandomGoal(C, idx, 1, movableWalls);
  }
  
  // Add blocked goals (numWalls=-1)
  for(int i = 0; i < numBlockedGoals; i++) {
    char idx = 'A' + numRegularGoals + i;
    addRandomGoal(C, idx, -1, movableWalls);
  }
  // addRandomGoal(C, 'D', 1);
  
  // Create new lgp config from scratch instead of loading and modifying
  rai::Graph lgpConfig;
  
  // Add fol path (absolute) - go from sourceDir up to problemGenerators, then to fol
  // std::string problemGeneratorsDir = sourceDir.substr(0, sourceDir.find_last_of("/\\"));
  // rai::String absFolPath = STRING(problemGeneratorsDir << "/fol/pnp-byTouch-multiGoal.fol");
  cout << "Source directory: " << sourceDir << endl;
  lgpConfig.add<rai::FileToken>("fol", "../../fol/pnp-byTouch-multiGoal.fol");
  cout << "Using fol file: " << lgpConfig.get<rai::FileToken>("fol").fullPath() << endl;
  
  // Build terminal rule
  rai::String terminal_rule = "";
  StringA explicitCollisions = {};
  for (rai::Frame *g : C.frames)
  {
    if (g->name.contains("wall"))
    {
      explicitCollisions.append("ego");
      explicitCollisions.append(g->name);
    }
  }
  for (rai::Frame *f : C.frames)
    if (f->name.contains("object") || f->name.contains("wallobj") || f->name.contains("wallgoal"))
    {
      if(!movableWalls && f->name.contains("wall")) continue;
      if (f->name.contains("object")){
      terminal_rule << "(on_goal " + f->name + ") ";
      explicitCollisions.append(f->name);
      explicitCollisions.append("ego");
      }
      for (rai::Frame *g : C.frames)
      {
        // cout << g->name << endl;
        if (g->name.contains("wall") && g->name != f->name)
        {
          rai::String name1;
          name1 << f->name;
          rai::String name2;
          name2 << g->name;
          explicitCollisions.append(name1);
          explicitCollisions.append(name2);
        }
      }
    }
  lgpConfig.set("terminal", terminal_rule);
  lgpConfig.set("coll", explicitCollisions);
  lgpConfig.add<bool>("genericCollisions", false);
  rai::FileToken file(STRING(sourceDir << "/temp_lgp_files/randomBlocks_temp" << id << ".lgp"));
  lgpConfig.write(file);
  C.coll_stepFcl();
  arr y, J;
  C.kinematicsPenetration(y, J);
  cout << "collision costs of config: " << y.scalar() << endl;

  // C.view(false);
  return C;
}

void createSingleWall(rai::Configuration &C, double x, double y, double wallOffset, str objName, int wallPos, 
                     double wallHeight, double wallThickness, double wallLength, bool moveableWalls)
{
  rai::Frame *wall = nullptr;
  str wallName;
  arr position;
  arr dimensions;

  if (wallPos == 0) { // North side (positive Y)
    wallName = STRING("wall" << objName << "_north");
    position = {x, y + wallOffset, wallHeight / 2};
    dimensions = {wallLength, wallThickness, wallHeight, 0.01};
  }
  else if (wallPos == 1) { // East side (positive X)
    wallName = STRING("wall" << objName << "_east");
    position = {x + wallOffset, y, wallHeight / 2};
    dimensions = {wallThickness, wallLength, wallHeight, 0.01};
  }
  else if (wallPos == 2) { // West side (negative X)
    wallName = STRING("wall" << objName << "_west");
    position = {x - wallOffset, y, wallHeight / 2};
    dimensions = {wallThickness, wallLength, wallHeight, 0.01};
  }
  else if (wallPos == 3) { // South side (negative Y)
    wallName = STRING("wall" << objName << "_south");
    position = {x, y - wallOffset, wallHeight / 2};
    dimensions = {wallLength, wallThickness, wallHeight, 0.01};
  }

  wall = C.addFrame(wallName, "floor");
  wall->setShape(rai::ShapeType::ST_ssBox, dimensions)
      .setColor({0.7, 0.7, 0.7})
      .setRelativePosition(position)
      .setContact(1);

  if (moveableWalls) {
    wall->setJoint(rai::JointType::JT_rigid);
    if (!wall->ats) {
      wall->ats = std::make_shared<rai::Graph>();
    }
    rai::Node_typed<rai::Graph> *logicObj = wall->ats->add<rai::Graph>("logical");
    logicObj->graph().add<bool>("is_object", true);
    logicObj->graph().add<bool>("is_obstacle", true);
  }

}

void addWallsAroundObject(rai::Configuration &C, double x, double y, double wallOffset, str objName, int numWalls, bool moveableWalls)
{
  double wallHeight = 0.3;
  double wallThickness = 0.15;
  double wallLength = 0.8;

  // Ensure numWalls is between 0 and 4
  numWalls = std::max(0, std::min(4, numWalls));

  // Randomly select numWalls out of 4 possible positions (0=north, 1=east, 2=west, 3=south)
  rai::Array<int> wallPositions = {0, 1, 2, 3};

  // Shuffle the positions and take the first numWalls
  for (int i = 3; i > 0; i--)
  {
    int j = (int)(rnd.uni() * (i + 1));
    int temp = wallPositions(i);
    wallPositions(i) = wallPositions(j);
    wallPositions(j) = temp;
  }

  // Create walls at the first numWalls selected positions
  for (int i = 0; i < numWalls; i++)
  {
    createSingleWall(C, x, y, wallOffset, objName, wallPositions(i), 
                    wallHeight, wallThickness, wallLength, moveableWalls);
  }
}

rai::Frame *addRandomObject(rai::Configuration &C, char idx, bool movableWalls)
{
  // collect existing object and goal frames before creating new ones
  rai::Array<rai::Frame *> existingFrames;
  for (rai::Frame *f : C.frames)
  {
    if (f->name.contains("object") || f->name.contains("goal") || f->name.contains("ego"))
    {
      existingFrames.append(f);
    }
  }

  // create object and position it to avoid existing objects and goals
  rai::Frame *obj = C.addFrame(STRING("object" << idx), "floor");
  double x, y;
  while (true)
  {
    bool tooClose = false;
    x = rnd.uni(-3.7, 3.7);
    y = rnd.uni(-3.7, 3.7);
    for (rai::Frame *f : existingFrames)
    {
      if (length(f->getPosition() - arr{x, y, 0.2}) < 2.0)
      {
        tooClose = true;
        break;
      }
    }
    if (!tooClose)
      break;
  }

  obj->setShape(rai::ShapeType::ST_ssBox,
                {0.3, 0.3, 0.3, 0.02})
      .setColor({0, 0, 1.0})
      .setRelativePosition({x, y, 0.2})
      .setJoint(rai::JointType::JT_rigid)
      .setContact(1);

  if (!obj->ats)
  {
    obj->ats = std::make_shared<rai::Graph>();
  }
  rai::Node_typed<rai::Graph> *logicObj = obj->ats->add<rai::Graph>("logical");
  logicObj->graph().add<bool>("is_object", true);

  // add walls around the object
  addWallsAroundObject(C, x, y, 0.45, STRING("obj" << idx), 2, movableWalls);

  return obj;
}

void addRandomGoal(rai::Configuration &C, char idx, int numWalls, bool movableWalls)
{
  // collect existing object and goal frames before creating new goal
  rai::Array<rai::Frame *> existingFrames;
  for (rai::Frame *f : C.frames)
  {
    if (f->name.contains("object") || f->name.contains("goal") || f->name.contains("ego"))
    {
      existingFrames.append(f);
    }
  }

  rai::Frame *goal = C.addFrame(STRING("goal" << idx), "floor");
  double x, y;

  // make sure the goal is not too close to objects and other goals
  while (true)
  {
    bool tooClose = false;
    x = rnd.uni(-3.7, 3.7);
    y = rnd.uni(-3.7, 3.7);
    for (rai::Frame *f : existingFrames)
    {
      if (length(f->getPosition() - arr{x, y, 0}) < 2)
      {
        tooClose = true;
        break;
      }
    }
    if (!tooClose)
      break;
  }
 
  // Apply modifications directly on this frame
  goal->setShape(rai::ShapeType::ST_ssBox,
                 {0.5, 0.5, 0.1, 0.01})
      .setColor({1., .3, .3}) 
      .setRelativePosition({x, y, 0}) 
      .setContact(0);

  if(numWalls>0) addWallsAroundObject(C, x, y, 0.3, STRING("goal" << idx), numWalls, movableWalls);
  else{
    goal->setShape(rai::ShapeType::ST_ssBox, {0.32, 0.32, 0.1, 0.01});
    // Add frame of obstacle on top of the goal
    rai::Frame *obstacle = C.addFrame(STRING("wallobj" << idx), goal->name);
    obstacle->setShape(rai::ShapeType::ST_ssBox, {0.3, 0.3, 0.3, 0.02})
        .setColor({0.7, 0.7, 0.7})
        .setRelativePosition({0, 0, 0.2})
        .setJoint(rai::JointType::JT_rigid) 
        .setContact(1);
    if (!obstacle->ats)
    {
      obstacle->ats = std::make_shared<rai::Graph>();
    }
    rai::Node_typed<rai::Graph> *logicObj = obstacle->ats->add<rai::Graph>("logical");
    logicObj->graph().add<bool>("is_object", true);
    logicObj->graph().add<bool>("is_obstacle", true);
    logicObj->graph().add<bool>("on_goal", true);
    logicObj->graph().add<bool>("busy", true);
    logicObj->graph().add<bool>("on", true);
  }

  // Make sure the ats graph exists
  if (!goal->ats)
  {
    goal->ats = std::make_shared<rai::Graph>();
  }

  rai::Node_typed<rai::Graph> *logic = goal->ats->add<rai::Graph>("logical");
  logic->graph().add<bool>("is_goal", true);
  logic->graph().add<bool>("is_place", true);
  if(numWalls <0) logic->graph().add<bool>("busy", true);
}