#include <Core/array.h>
#include <Core/graph.h>
#include <Gui/opengl.h>
#include <Kin/kin.h>
#include <Kin/proxy.h>
#include <Kin/viewer.h>
#include <Logic/fol.h>
#include <Logic/folWorld.h>

#include <KOMO/komo.h>
#include <LGP/LGP_node.h>
#include <LGP/LGP_tree.h>

#include <map>
#include <set>
#include <string>

#include "json.hpp"

const char *BoundTypeNames[] = {"BD_symbolic", "BD_pose",    "BD_seq",
                                "BD_path",     "BD_seqPath", "BD_max"};

// for convenience
using json = nlohmann::json;

class Cache {
private:
  std::set<std::string> feasible_prefixes;
  std::set<std::string> infeasible_prefixes;
  std::map<int, int> count_feasible_by_len;
  std::map<int, int> count_infeasible_by_len;

public:
  Cache() {}
  ~Cache() {}

  // check if prefix is in cache. Return 0 if not found, 1 if feasible, -1 if
  // infeasible
  int check_feasibility(rai::String prefixS) const {
    std::string prefix = std::string(prefixS.p);
    if (feasible_prefixes.find(prefix) != feasible_prefixes.end()) {
      return 1;
    } else if (infeasible_prefixes.find(prefix) != infeasible_prefixes.end()) {
      return -1;
    } else {
      return 0;
    }
  }

  // add prefix to cache as feasible (if feasible=1) or infeasible (if feasible
  // = -1)
  void add_prefix(rai::String prefixS, int prefix_len, int feasible) {
    std::string prefix = std::string(prefixS.p);
    if (feasible == 1) {
      feasible_prefixes.insert(prefix);
      count_feasible_by_len[prefix_len] = count_feasible_by_len[prefix_len] + 1;
    } else if (feasible == -1) {
      infeasible_prefixes.insert(prefix);
      count_infeasible_by_len[prefix_len] =
          count_infeasible_by_len[prefix_len] + 1;
    }
  }

  // go over feasible prefixes (if feasible = 1) or infeasible prefixes (if
  // feasible = -1), and count how many have the given prefix
  int count_by_prefix(rai::String prefix, int feasible) const {
    std::set<std::string>::const_iterator b, e;
    if (feasible == 1) {
      b = feasible_prefixes.begin();
      e = feasible_prefixes.end();
    } else if (feasible == -1) {
      b = infeasible_prefixes.begin();
      e = infeasible_prefixes.end();
    }
    int count = 0;
    while (b != e) {
      rai::String c_prefix(*b);
      if (c_prefix.startsWith(prefix)) {
        count++;
      }
      b++;
    }
    return count;
  }

  // return count of feasible prefixes (if feasible = 1) or infeasible prefixes
  // (if feasible = -1) by length
  int count_by_len(int len, int feasible) {
    if (feasible == 1) {
      return count_feasible_by_len[len];
    } else if (feasible == -1) {
      return count_infeasible_by_len[len];
    } else {
      return 0;
    }
  }

  int count(int feasible) const {
    if (feasible == 1) {
      return feasible_prefixes.size();
    } else if (feasible == -1) {
      return infeasible_prefixes.size();
    } else {
      return 0;
    }
  }

  void set(const std::set<std::string> &t_feasible_prefixes,
           const std::set<std::string> &t_infeasible_prefixes) {
    feasible_prefixes = t_feasible_prefixes;
    infeasible_prefixes = t_infeasible_prefixes;

    for (auto &p : feasible_prefixes) {
      size_t prefix_len = std::count(
          p.begin(), p.end(), '('); // pretty stupid way to find length of plan
      count_feasible_by_len[prefix_len] = count_feasible_by_len[prefix_len] + 1;
    }

    for (auto &p : infeasible_prefixes) {
      size_t prefix_len = std::count(
          p.begin(), p.end(), '('); // pretty stupid way to find length of plan
      count_infeasible_by_len[prefix_len] =
          count_infeasible_by_len[prefix_len] + 1;
    }
  }

  void read() {
    json j;
    j << FILE("cache.json");
    feasible_prefixes.insert(j["feasible_prefixes"].begin(),
                             j["feasible_prefixes"].end());
    infeasible_prefixes.insert(j["infeasible_prefixes"].begin(),
                               j["infeasible_prefixes"].end());
    count_feasible_by_len.insert(j["count_feasible_by_len"].begin(),
                                 j["count_feasible_by_len"].end());
    count_infeasible_by_len.insert(j["count_infeasible_by_len"].begin(),
                                   j["count_infeasible_by_len"].end());
    cout << "Got " << feasible_prefixes.size() << " feasible prefixes" << endl;
    cout << "Got " << infeasible_prefixes.size() << " infeasible prefixes"
         << endl;
  }

  void write(const char *filename_out = "cache.json") const {
    // feasible_prefixes.write(FILE("cache.feasible.txt"));
    // infeasible_prefixes.write(FILE("cache.infeasible.txt"));
    json j;
    j["feasible_prefixes"] = feasible_prefixes;
    j["infeasible_prefixes"] = infeasible_prefixes;
    j["count_feasible_by_len"] = count_feasible_by_len;
    j["count_infeasible_by_len"] = count_infeasible_by_len;
    j >> FILE(filename_out);
  }
};

class MDP {
private:
  uint max_len;
  rai::Array<double> val;
  rai::Array<uint> decision;
  Cache &cache;
  rai::LGP_Node *plan_node;
  rai::LGP_NodeL path;

public:
  MDP(uint max_prefix_len, Cache &cache, rai::LGP_Node *plan_node)
      : max_len(max_prefix_len + 1), val(rai::Array<double>(max_len, max_len)),
        decision(rai::Array<uint>(max_len, max_len)), cache(cache),
        plan_node(plan_node) {
    path = plan_node->getTreePath();
  }
  ~MDP() {}

  rai::String get_prefix(int prefix_len) {
    return path(prefix_len)->getTreePathString();
  }

  // return reward for conflict of given length
  double R(int prefix_len) {
    // TODO: count plans only, not prefixes
    rai::String prefix = get_prefix(prefix_len);
    int feasibleP = cache.count_by_prefix(prefix, 1);
    int infeasibleP = cache.count_by_prefix(prefix, -1);
    int feasible = cache.count(1);
    int infeasible = cache.count(-1);

    // use at least 1 in the numerator
    // if ((feasibleP + infeasibleP) == 0) {
    //  feasibleP = 1;
    //}

    // cout << "Counts for" << endl;
    // cout << prefix_len << "(" << prefix << ")" << endl;
    // cout << feasibleP << " | " << infeasibleP << " / " << feasible << " | "
    // << infeasible << endl;

    return 1.0 * (feasibleP + infeasibleP + 1) / (feasible + infeasible + 1);
  }

  double Pf(int prefix_len) {
    int feasibleC = cache.count_by_len(prefix_len, 1);
    int infeasibleC = cache.count_by_len(prefix_len, -1);

    // cout << "For len: " << prefix_len << " got counts " << feasibleC << " | "
    // << infeasibleC << endl;

    double p = (feasibleC + 1) * 1.0 / (feasibleC + infeasibleC + 2);
    // if (p < 0.1) {
    //   return 0.1;
    // } else if (p > 0.9) {
    //   return 0.9;
    // } else {
    return p;
    //}
  }

  void compute_values() {
    val.setId();
    decision.setZero();
    // initialize values of terminal states
    for (uint i = 0; i < max_len; i++) {
      val(i, i) = R(i);
    }

    // compute optimal values for states (lb, lb+d), in increasing order of d
    for (uint d = 1; d < max_len; d++) {
      for (uint lb = 0; lb + d < max_len; lb++) {
        uint ub = lb + d;

        // compute value of stopping
        double R_stop = val(ub, ub);
        decision(lb, ub) = 0;
        val(lb, ub) = R_stop;

        // compute value of continuing search
        for (uint mid = lb; mid < ub; mid++) {
          // int mid = (lb + ub) /  2;
          double pf = Pf(mid);
          double R_continue =
              (pf * val(mid, ub)) + ((1.0 - pf) * (val(lb, mid)));
          // record optimum decision
          if (R_continue > val(lb, ub)) {
            decision(lb, ub) = mid;
            val(lb, ub) = R_continue;
          }
        }
      }
    }
  }

  uint get_decision(uint lb, uint ub) { return decision(lb, ub); }

  void dump() {
    cout << "Value: " << endl;
    cout << val << endl;
    cout << "Decision: " << endl;
    cout << decision << endl;
  }
};

//===========================================================================

void generatePDDL(rai::LGP_Tree &lgp) {
  cout << "Generating PDDL Files" << endl;

  lgp.fol.writePDDLfiles("kaka");
}

//===========================================================================

bool readPlan(rai::LGP_Tree &lgp, rai::String planFile,
              double minConflictGapThreshold, bool use_dd_mdp, bool display_feasible,  int verbosity) {


  bool path_only_goal_node = true;

  rai::String plan(FILE(planFile));
  Cache cache;
  cache.read();

  // cut the last line comment with ';'
  uint i = plan.N;
  for (; i--;)
    if (plan(i) == ';')
      break;
  plan.resize(i, true);

  cout << "FOUND PLAN: " << plan << endl;

  rai::String file = rai::raiPath("test/Logic/player/pnp.g");
  // rai::String file = "../../LGP/pickAndPlace/fol-pnp-switch.g";
  // world.addDecisionSequence(plan);

  // lgp.inspectSequence(plan);

  rai::LGP_Node *node = lgp.walkToNode(plan);
  node->checkConsistency();
  cout << "got node " << node->getTreePathString() << endl;

  MDP mdp(node->step, cache, node);

  if (use_dd_mdp) {
    cout << "Initializing MDP for length " << node->step << endl;
    mdp.compute_values();
    mdp.dump();
  }

  bool feasible = true;

  int cache_check = cache.check_feasibility(node->getTreePathString());
  if (cache_check == 1) {
    cout << "Found answer in cache: " << cache_check << endl;
    feasible = true;
  } else if (cache_check == -1) {
    cout << "Found answer in cache: " << cache_check << endl;
    feasible = false;
  } else {
    cout << "Not found in cache, checking feasibility" << endl;
    uint lb = 1;          // minimal node which might be infeasible
    uint ub = node->step; // maximal node which is infeasible

    // check pose bound for each node along path
    rai::LGP_Node *path_node = node;

    bool pose_incremental = true;

    if (pose_incremental) {
      auto path = node->getTreePath();
      int N = path.N;
      int i = 1;

      // rai::LGP_Node* nodeq = node;
      // while (nodeq->parent != nullptr) {
      //   cout << "current node " << nodeq->getTreePathString()  << endl;
      //   nodeq =  nodeq->parent;   ;
      // }
      // cout << "path length" << endl;
      // cout << node->getTreePath().N << std::endl;

      // Path length = # actions + 1
      // Example:
      // current node (pick block1 table l_gripper) (pick block2 table
      // r_gripper) (place block2 r_gripper block1) (place block1 l_ gripper
      // box1) current node (pick block1 table l_gripper) (pick block2 table
      // r_gripper) (place block2 r_gripper block1) current node (pick block1
      // table l_gripper) (pick block2 table r_gripper) current node (pick
      // block1 table l_gripper) path length
      // 5

      while (i < N) {
        path_node = path(i);
        cout << "current node " << path_node->getTreePathString() << endl;
        int path_node_cache_check =
            cache.check_feasibility(path_node->getTreePathString());
        if (path_node_cache_check == -1) {
          feasible = false;
          ub = path_node
                   ->step; // update upper bound according to path consistency
          cout << "Found infeasible prefix in cache, updated UB: "
               << path_node->getTreePathString() << endl;
        } else if (path_node_cache_check == 1) {
          if (lb == 1) {
            lb = path_node->step + 1;
            cout << "Found feasible prefix in cache, updating LB: "
                 << path_node->getTreePathString() << endl;
          }
        } else {
          cout << "CHECK_POSE" << endl;
          path_node->optBound(rai::BD_pose, true, verbosity);
          if (!path_node->feasible(rai::BD_pose)) {
            feasible = false;
            ub = path_node
                     ->step; // update upper bound according to path consistency
            cout << "Pose not feasible, adding to cache: "
                 << path_node->getTreePathString() << endl;
            cache.add_prefix(path_node->getTreePathString(), path_node->step,
                             -1);
            break;
          }
        }
        i++;
      }

    } else {
      while (path_node->parent != nullptr) {
        cout << "current node " << path_node->getTreePathString() << endl;
        int path_node_cache_check =
            cache.check_feasibility(path_node->getTreePathString());
        if (path_node_cache_check == -1) {
          feasible = false;
          ub = path_node
                   ->step; // update upper bound according to path consistency
          cout << "Found infeasible prefix in cache, updated UB: "
               << path_node->getTreePathString() << endl;
        } else if (path_node_cache_check == 1) {
          if (lb == 1) {
            lb = path_node->step + 1;
            cout << "Found feasible prefix in cache, updating LB: "
                 << path_node->getTreePathString() << endl;
          }
        } else {
          cout << "CHECK_POSE" << endl;
          path_node->optBound(rai::BD_pose, true, verbosity);
          if (!path_node->feasible(rai::BD_pose)) {
            feasible = false;
            ub = path_node
                     ->step; // update upper bound according to path consistency
            cout << "Pose not feasible, adding to cache: "
                 << path_node->getTreePathString() << endl;
            cache.add_prefix(path_node->getTreePathString(), path_node->step,
                             -1);
            break;
          }
        }
        path_node = path_node->parent;
      }
    }

    // check sequence bound for final node
    if (feasible) {
      cout << "CHECK_SEQ" << endl;
      node->optBound(rai::BD_seq, true, verbosity);
      feasible = node->feasible(rai::BD_seq);
      if (!feasible) {
        cout << "Seq not feasible, adding to cache" << endl;
        cache.add_prefix(node->getTreePathString(), node->step, -1);
      }
    }

    // check path bound for final node
    if (feasible) {
      cout << "CHECK_PATH" << endl;
      node->optBound(rai::BD_path, true, verbosity);
      feasible = node->feasible(rai::BD_path);
      if (!feasible) {
        cout << "Path not feasible, adding to cache" << endl;
        cache.add_prefix(node->getTreePathString(), node->step, -1);
      } else {
        cout << "feasible plan! lets add to cache" << endl;
        cache.add_prefix(node->getTreePathString(), node->step, 1);
      }
    }

    // Find minimal conflict
    if (!feasible) {
      rai::LGP_NodeL path = node->getTreePath();

      cout << "starting at depth: " << ub << endl;
      while ((ub - lb) > minConflictGapThreshold) {
        uint mid = (lb + ub) / 2;
        if (use_dd_mdp) {
          mid = mdp.get_decision(lb, ub);
          cout << "MDP decision for (" << lb << ", " << ub << ") = " << mid
               << endl;
          if (mid == 0) {
            cout << "MDP decided to stop" << endl;
            break;
          }
        }

        // uint mid = (lb + ub) /  2;
        cout << "Range [" << lb << "," << ub << "] - checking " << mid << endl;

        bool feasible = false;

        rai::LGP_Node &mid_node = *path(mid);
        int mid_node_cache_check =
            cache.check_feasibility(mid_node.getTreePathString());
        if (mid_node_cache_check == 1) {
          feasible = true;
        } else if (mid_node_cache_check == -1) {
          feasible = false;
        } else {
          if (mid_node.feasible(rai::BD_pose)) {
            cout << "CHECK_SEQ" << endl;
            mid_node.optBound(rai::BD_seq, true, verbosity);
            if (mid_node.feasible(rai::BD_seq)) {
              if (!path_only_goal_node)
              {
                cout << "CHECK_PATH" << endl;
                mid_node.optBound(rai::BD_path, true, verbosity);
                if (mid_node.feasible(rai::BD_path)) {
                  feasible = true;
                }
              }
              else {
                feasible = true;
              }
            }
          }
        }
        if (feasible) {
          cout << "feasible" << endl;
          // add all prefixes up to mid, as they are all feasible
          for (uint prefix_len = 1; prefix_len <= mid; prefix_len++) {
            cache.add_prefix(path(prefix_len)->getTreePathString(),
                             path(prefix_len)->step, 1);
          }
          lb = mid + 1;
        } else {
          cout << "infeasible" << endl;
          cache.add_prefix(mid_node.getTreePathString(), mid_node.step, -1);
          ub = mid;
        }
      }
      rai::LGP_Node &min_conflict_node = *path(ub);
      cout << "min conflict: " << min_conflict_node.getTreePathString() << endl;
    }
  }

  cache.write();

  if (feasible && display_feasible) {
    rai::ConfigurationViewer K;
    node->displayBound(K, rai::BD_path);
  }

  return feasible;
}

bool solve_lgp(rai::LGP_Tree &lgp, int steps, rai::BoundType bound) {

  std::cout << "SOLVING LGP with bound " << BoundTypeNames[bound] << std::endl;
  lgp.bound_before_expand = bound;
  lgp.run(steps);
  for (auto *s : lgp.solutions.set()()) {
    cout << "SOLUTION:\n";
    s->write(cout);
    cout << endl;
  }

  // lets write down the results

  std::cout << "infeas " << std::endl;
  for (auto &s : lgp.infeasible_prefixes) {
    std::cout << s << std::endl;
  }

  std::cout << "feas " << std::endl;
  for (auto &s : lgp.feasible_prefixes) {
    std::cout << s << std::endl;
  }

  Cache cache;
  cache.set(lgp.feasible_prefixes, lgp.infeasible_prefixes);
  cache.write();

  std::stringstream out;
  std::cout << "RESULTS" << std::endl;
  out << "TIME= " << rai::cpuTime() << " TIME= " << rai::COUNT_time
      << " KIN= " << rai::COUNT_kin << " TREE= " << rai::COUNT_node
      << " POSE= " << rai::COUNT_opt(rai::BD_pose)
      << " SEQ= " << rai::COUNT_opt(rai::BD_seq) << " PATH= "
      << rai::COUNT_opt(rai::BD_path) + rai::COUNT_opt(rai::BD_seqPath)
      << std::endl;
  std::cout << out.str() << std::endl;

  return true;
}

//===========================================================================

// LGP baseline
// ./x.exe -mode lgp -bound 0
// ./x.exe -mode lgp -bound 2
int main(int argn, char **argv) {
  rai::initCmdLine(argn, argv);
  rnd.clockSeed();

  //-- we need both, a logic file (<-> domain) and a geometric configuration
  // file (<-> problem)
  rai::String folFile =
      rai::getParameter<rai::String>("folFile", STRING("none"));
  rai::String confFile =
      rai::getParameter<rai::String>("confFile", STRING("none"));
  int verbosity = rai::getParameter<int>("verbosity", 0);

  if (rai::argc >= 2 && rai::argv[1][0] != '-') {
    folFile = rai::argv[1];
    if (rai::argc >= 3 && rai::argv[2][0] != '-')
      confFile = rai::argv[2];
  }

  LOG(0) << "using fol-file '" << folFile << "'" << endl;
  LOG(0) << "using conf-file '" << confFile << "'" << endl;

  //-- create configuration
  rai::Configuration C;
  C.addFile(confFile);

  //-- create LGP tree
  rai::LGP_Tree lgp(C, folFile);

  //-- we need a goal - the generic domain logic does not define one
  lgp.fol.addTerminalRule(rai::getParameter<rai::String>("goal", STRING("{}")));

  if (rai::getParameter<rai::String>("mode", "") == "gen") {
    generatePDDL(lgp);
  } else if (rai::getParameter<rai::String>("mode", "") == "sol") {
    rai::system(
        "~/git/downward/fast-downward.py --plan-file kaka.sas_plan "
        "kaka.domain.pddl kaka.problem.pddl --landmarks 'lm=lm_hm(m=2)' "
        "--search 'astar(lmcount(lm,admissible=True,dump=True))'");
  } else if (rai::getParameter<rai::String>("mode", "") == "lgp") {
    // solve with lgp
    double steps = rai::getParameter<double>("steps", 10000);
    int bound = int(rai::getParameter<double>("bound", 0));
    solve_lgp(lgp, int(steps), static_cast<rai::BoundType>(bound));
  } else if (rai::getParameter<rai::String>("mode", "") == "read") {

    rai::String planFile =
        rai::getParameter<rai::String>("planFile", STRING("none"));
    double minConflictGapThreshold =
        rai::getParameter<double>("minConflictGapThreshold", 0);
    bool use_dd_mdp = rai::getParameter<bool>("useDDMDP", false);

    bool display = rai::getParameter<bool>("display", false );
    bool feasible =
        readPlan(lgp, planFile, minConflictGapThreshold, use_dd_mdp, display , verbosity);
    cout << "Final verdict - feasible: " << feasible << endl;
  }
}
