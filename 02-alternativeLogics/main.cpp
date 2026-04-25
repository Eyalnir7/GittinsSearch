#include <LGP/LGP_tree.h>

//===========================================================================

void exploreLGP(){
  
  rai::KinematicWorld C;
  C.addFile(rai::getParameter<rai::String>("model"));
  
  LGP_Tree lgp(C, rai::getParameter<rai::String>("fol"));

  if(rai::checkParameter<int>("tree"))
    lgp.buildTree(rai::getParameter<int>("tree"));

  if(rai::checkParameter<bool>("dot")) 
    lgp.displayTreeUsingDot();

  if(rai::checkParameter<bool>("play"))
    lgp.player();

}

//===========================================================================

void test(){

  rai::KinematicWorld C;
  C.addFile(rai::getParameter<rai::String>("model"));

  LGP_Tree lgp(C, rai::getParameter<rai::String>("fol"));

  lgp.fol.addTerminalRule("(on tray obj0)");
  lgp.fol.writePDDLfiles("z.");

  rai::wait();

  LGP_Node *node = lgp.walkToNode({"(pick table1 pr2R obj0) (place pr2R obj0 tray)"});
//  LGP_Node *node = lgp.walkToNode({"(linkBreak pr2R obj0 table1 obj0) (linkBreak tray obj0 pr2R obj0)"});

  BoundType bound = BD_path;
  node->optBound(bound, false, 4);
  while(node->komoProblem(bound)->displayTrajectory(.1, true, false));


}

//===========================================================================

int main(int argc,char** argv){
  rai::initCmdLine(argc,argv);

//  exploreLGP();
  test();

  return 0;
}

