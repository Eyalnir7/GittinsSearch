#pragma once

#include <Kin/frame.h>
#include <Core/util.h>

// Forward declarations
namespace rai {
    class Configuration;
    class Frame;
    class Rnd;
}

// External random number generator (defined in main.cpp)
extern rai::Rnd rnd;

// Function declarations
rai::Configuration getRandomConfiguration(bool movableWalls, int id, bool logVerbose=false);
void addRandomGoal(rai::Configuration &C, char idx, int numWalls = 2, bool movableWalls = false);
rai::Frame *addRandomObject(rai::Configuration &C, char idx, bool movableWalls);
void createSingleWall(rai::Configuration &C, double x, double y, double wallOffset, str objName, int wallPos, 
                     double wallHeight, double wallThickness, double wallLength, bool moveableWalls);
void addWallsAroundObject(rai::Configuration &C, double x, double y, double wallOffset, str objName, int numWalls = 2, bool moveableWalls = false);