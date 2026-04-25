#include "ConfigurationGenerator.h"
#include <Kin/frame.h>
#include <Kin/kin.h>
#include <Core/graph.h>
#include <Core/util.h>

rai::Configuration getRandomConfiguration(bool movableWalls, int id, bool logVerbose)
{
    rnd.seed_random();
  str problem = "2blocks";

  str lgpFile = problem + ".lgp";
  str confFile = problem + ".g";
  LOG(0) << "using lgpFile: '" << lgpFile << "'";
  LOG(0) << "using confFile: '" << confFile << "'";

  rai::Configuration C;
  C.addFile(confFile);

  rai::Frame *objA = addRandomObject(C, 'A', movableWalls);
  addRandomGoal(C, 'A', 1, movableWalls);

  rai::Frame *objB = addRandomObject(C, 'B', movableWalls);
  addRandomGoal(C, 'B', 1, movableWalls);

  //with probability 0.5 add a third object
  if(rnd.uni()<0.5){
    rai::Frame* objC = addRandomObject(C, 'C', movableWalls);
    addRandomGoal(C, 'E', -1, movableWalls);
  }
  addRandomGoal(C, 'C', 1, true);
  addRandomGoal(C, 'D', -1, movableWalls);
  // addRandomGoal(C, 'D', 1);
  rai::Graph lgpConfig("2blocks.lgp");
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
  if(logVerbose){
    rai::FileToken file(STRING("temp_lgp_files/2blocks-temp" << id << ".lgp"));
    lgpConfig.write(file);
  }
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
    // logicObj->graph().add<bool>("busy", true);
    // logicObj->graph().add<bool>("on", true);
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