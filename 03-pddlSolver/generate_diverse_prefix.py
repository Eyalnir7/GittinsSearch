import argparse
import subprocess
import logging
import shutil
import os.path
import os
import json
import glob
import math
import scipy
import time
from datetime import datetime
import random

class Cache:
    """This class represents the cache that is shared with the LGP feasibility checking code. Implemented read/write to JSON file."""
    def __init__(self):
        """Initialize cache object"""
        self.data = {}
        self.data["count_feasible_by_len"] = []
        self.data["count_infeasible_by_len"] = []
        self.data["feasible_prefixes"] = []
        self.data["infeasible_prefixes"] = []

    def write(self, fname):
        """Write cache to JSON file"""
        with open(fname, "w") as f:
            json.dump(self.data, f)

    def read(self, fname):
        """Read cache from JSON file"""
        with open(fname, "r") as f:
            self.data = json.load(f)

def jaccard_similarity(set1, set2):
    intersection = len(list(set(set1).intersection(set2)))
    union = (len(set1) + len(set2)) - intersection
    return float(intersection) / union

class Plan:
    """This class represents a plan, which can be loaded from a string (or from a file using read_from_file)"""
    def __init__(self, plan_string="", actions=[]):
        """Initialize plan with given plan string"""
        if plan_string != "" and actions != []:
            raise AttributeError
        if plan_string != "":
            self.plan_string = plan_string
            self.process_plan_string()
        elif actions != []:
            self.actions = actions
        else: 
            self.plan_string = ""
            self.actions = []


    def __len__(self):
        return len( self.actions)

    def __getitem__(self, key):
        #print("key is {}".format(key))
        if isinstance(key, slice):
          start, stop, step = key.indices(len(self))
          return  Plan(actions=[self.actions[i] for i in range(start, stop, step)])
        elif isinstance(key, int):
          return self.actions[key]
        elif isinstance(key, tuple):
          raise NotImplementedError
        else:
          raise TypeError



    def process_plan_string(self):
        """Helper function to split plan string into actions and remove comments"""
        ind = self.plan_string.find(";")
        if ind >= 0:
            self.plan_string = self.plan_string[:ind]

        self.actions = list(map(
            lambda a: a.strip() + ")", 
                filter(lambda a: len(a.strip()) > 0,
                        self.plan_string.split(")"))))        

    def read_from_file(self, filename):
        """Read plan from file"""
        with open(filename) as f:
            self.plan_string = f.read()
        self.process_plan_string()

    def write_to_file(self, filename):
        """Write plan to file"""
        with open(filename, "w") as f:     
            for a in self.actions:
                f.write(a + "\n")            

    def tokenize(self):
        """return plan as List[List[str]]"""
        out = []
        for a in self.actions:
            out.append(a[a.find("(")+1:a.find(")")].split())
        return out

    def checkLoop1(self):
        """return last index of infeasible prefix, -1 is feasible""" 
        out = self.tokenize()
        for i, a  in enumerate(out):
            if a[0] == 'place' and a[1] == a[3]:
                return i
        return -1


    def __repr__(self):
        """String representation"""
        return "|".join(self.actions)


    def similarity(self, plan2, gamma=1.0):
        """Similarity measure"""
        if gamma == 1.0:
            return jaccard_similarity(self.actions, plan2.actions)
        else:
            minlen = min(len(self.actions), len(plan2.actions))
            total = 0.0
            norm = 0.0
            for i in range(1, minlen + 1):
                s = jaccard_similarity(self.actions[:i], plan2.actions[:i])
                total = total + (s * (gamma ** i))
                norm = norm + (gamma ** i)            
            return total / norm

    def common_prefix(self, plan2):
        return os.path.commonprefix([self.actions, plan2.actions])


        #return w1 * jaccard_similarity(self.actions, plan2.actions) + w2 * jaccard_similarity(self.get_partially_ordered_pairs(), plan2.get_partially_ordered_pairs()) 

    #def get_partially_ordered_pairs(self):
    #    l = []
    #    for i, a1 in enumerate(self.actions):
    #        for a2 in self.actions[i+1:]:
    #            l.append( (a1, a2) )
    #    return l


def lowercase_bool(val):
    if val:
        return "true"
    else:
        return "false"

class LGPDiverseSolver:
    """This class implements the LGP Diverse Planning Solver"""
    def __init__(self, args):
        """Initialize solver with command line args"""
        self.args = args
        random.seed(self.args.seed)

        if self.args.conflict == "eager":
            self.args.minConflictGapThreshold = 0
            self.args.useDDMDP = False
        elif self.args.conflict == "lazy":
            self.args.minConflictGapThreshold = 9999
            self.args.useDDMDP = False
        elif self.args.conflict == "mr":
            self.args.minConflictGapThreshold = 0
            self.args.useDDMDP = True

        self.num_generated_plans = 0
        self.num_tested_plans = 0
        self.num_feasibility_checks_pose = 0
        self.num_feasibility_checks_seq = 0
        self.num_feasibility_checks_path = 0
        self.solution = []
        self.solved = False
        self.plans = {}
        self.tested_plans = {}         
        self.eliminated_plans = {}
        self.cache = Cache()
        self.initializeFiles()

    def initializeFiles(self):
        """Initialize files - make sure directories for plans are empty, generate an empty cache"""
        # Clean up everything beforehand
        shutil.rmtree(self.args.conflictsDir, ignore_errors=True)
        os.makedirs(self.args.conflictsDir)
        os.makedirs( os.path.join(self.args.conflictsDir, "original"))

        shutil.rmtree(self.args.debugDir, ignore_errors=True)
        os.makedirs(self.args.debugDir)
        
        logfiles = glob.glob('*.log')
        for zfile in logfiles:
            shutil.rmtree(zfile, ignore_errors=True)



        zfiles = glob.glob('z.*')
        for zfile in zfiles:
            shutil.rmtree(zfile, ignore_errors=True)

        files_to_remove = ["kaka.domain.pddl", "kaka.problem.pddl", "output.sas", "reformulated_output.sas"]
        for file_to_remove in files_to_remove:
            shutil.rmtree(file_to_remove, ignore_errors=True)
        

        # Initialize cache
        self.cache.write(self.args.cache)


    def callExternalCommand(self, cmd):
        """Helper function to call an external command"""        
        with open("err.log", "w") as ferr:
            logging.debug( "run command: {}".format(cmd))
            ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=ferr)
            logging.debug( "got output: {}".format(ret.stdout.decode("utf-8")) )            
        return ret

    def callLGP(self, mode, additional_args = []):
        """Call external command of LGP binary"""
        return self.callExternalCommand([self.args.LGPExecutable, "-mode", mode, "-cfg", self.args.config , "-display" , str(self.args.display)] + additional_args)

    def callLGPGen(self):
        """Call LGP PDDL Generation"""
        return self.callLGP("gen")

    def callLGPCheck(self, plan_fname):          
        """Call LGP Geometric Feasibility Checking"""        
        ret = self.callLGP("read", 
            ["-planFile", plan_fname,
            "-minConflictGapThreshold", str(self.args.minConflictGapThreshold),
            "-useDDMDP", lowercase_bool(self.args.useDDMDP)])

        pVerdict = "Final verdict - feasible:"
        pConflict = "min conflict:"

        feasible = 0
        conflict = None

        out = ret.stdout.decode("utf-8")
        plan_fname_base = plan_fname[plan_fname.rfind("/")+1:] 
        filename = self.args.debugDir + "/" + "z.komo" + plan_fname_base
        with open(filename, 'w') as f:
            f.write(out)
        cmd_output_lines = out.splitlines()

        for line in cmd_output_lines:
            if line.startswith("CHECK_POSE"):
                self.num_feasibility_checks_pose = self.num_feasibility_checks_pose + 1
            if line.startswith("CHECK_SEQ"):
                self.num_feasibility_checks_seq = self.num_feasibility_checks_seq + 1
            if line.startswith("CHECK_PATH"):
                self.num_feasibility_checks_path = self.num_feasibility_checks_path + 1
            if line.startswith(pVerdict):                    
                feasible = int(line[len(pVerdict):])
                logging.debug("Feasible: " + str(feasible) + " | " + line)
            if line.startswith(pConflict):
                conflict = Plan(line[len(pConflict):])
                logging.debug("Conflict: " + str(conflict) + " | " + line)                    
        self.num_tested_plans = self.num_tested_plans + 1
        return (ret, feasible, conflict)



    def callReformulation(self):
        """Call Forbid Iterative to reformulate problem, avoiding conflicts and plans (in conflicts dir)"""
        if self.num_generated_plans == 0:
            return self.callExternalCommand([self.args.ForbidIterativeExecutable, "--build", self.args.ForbidIterativeBuild,
            "--translate", "kaka.domain.pddl","kaka.problem.pddl",
            "--translate-options", "--keep-unimportant-variable"])            
            # Make sure to keep unimportant variables, as they might be important in geometric layer
        else:
            return self.callExternalCommand([self.args.ForbidIterativeExecutable, "--build", self.args.ForbidIterativeBuild,
            "output.sas",
            "--search", 
            "forbid_iterative(reformulate = FORBID_MULTIPLE_PLAN_PREFIXES, change_operator_names=false, number_of_plans_to_read=" + str(self.num_generated_plans) + ",external_plans_path=" + self.args.conflictsDir + ")"])

    def callPlanner(self):
        """Call planner on reformulated problem. Copy plan to plans dir and conflicts dir, and add it to dictionary with plans"""
        sas_file = "reformulated_output.sas"
        if self.num_generated_plans == 0:
            sas_file = "output.sas"

        ret = self.callExternalCommand([self.args.ForbidIterativeExecutable, 
            "--alias", self.args.ForbidIterativePlannerAlias,
            "--build", self.args.ForbidIterativeBuild,
            sas_file            
            #"--heuristic", "hlm,hff=lm_ff_syn(lm_rhw(reasonable_orders=true,lm_cost_type=one),"
            #"transform=adapt_costs(one))",
            #"--search", "lazy_greedy([hff,hlm],preferred=[hff,hlm], cost_type=one,reopen_closed=false)"
            #"--symmetries", "sym=structural_symmetries(time_bound=0,search_symmetries=oss, stabilize_initial_state=false, keep_operator_symmetries=false)",
            #"--search", "astar(celmcut(), shortest=false, symmetries=sym)"
            ])

        out = ret.stdout.decode("utf-8")
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
        filename =  self.args.debugDir + "/" + "planner." + dt_string 
        with open(filename, "w") as f:
            f.write(out)
        
        

        # Plan will be written to conflicts dir, then overwritten if a better conflict is extracted
        for gen_plan in glob.glob("sas_plan*"):
            # TODO: why one? is this a bug? EREZ - no...
            self.num_generated_plans = self.num_generated_plans + 1
            plan_fname = self.args.basePlanFilename + "." + str(self.num_generated_plans)
            conflict_path = os.path.join(self.args.conflictsDir, plan_fname)
            shutil.move(gen_plan, conflict_path)
            p = Plan("")
            p.read_from_file(conflict_path)
            self.plans[plan_fname] = p

        return ret

    def generateDiversePlans(self, N):
        """Generate a set of N plans by calling reformulations and then planner"""
        for i in range(N):
            self.callReformulation()
            self.callPlanner()

    #TODO: always choose from N different plans - eliminate plans which are inconsistent with prefixes
    def chooseNextPlan(self):
        """Choose the next (untested) plan according to objective of avg. similarity to feasible prefixed - avg. similarity to infeasible prefixes"""
        best_plans = []
        best_obj = math.inf
        if self.args.novelty:
            for plan_fname, plan in self.plans.items():
                if plan_fname not in self.tested_plans.keys() and plan_fname not in self.eliminated_plans.keys():
                    #plan = self.plans[plan_fname]
                    
                    novelty = -math.inf
                    for tested_plan_fname in self.tested_plans.keys():
                        n1 = len(plan.common_prefix(self.plans[tested_plan_fname]))
                        if n1 > novelty:
                            novelty = n1
                    logging.debug(plan_fname + " novelty: " + str(novelty))
                    if novelty < best_obj:
                        best_obj = novelty
                        best_plans = [plan_fname]
                    elif novelty == best_obj:
                        best_plans.append(plan_fname)                    
        else:
            self.cache.read(self.args.cache)

            feasible_prefixes = list(map(lambda x: Plan(x), self.cache.data['feasible_prefixes']))
            infeasible_prefixes = list(map(lambda x: Plan(x), self.cache.data['infeasible_prefixes']))

            best_obj = -math.inf        

            for plan_fname in self.plans:
                if plan_fname not in self.tested_plans.keys() and plan_fname not in self.eliminated_plans.keys():
                    avg_sim_feas = 0.0
                    avg_sim_infeas = 0.0

                    p = self.plans[plan_fname]

                    if len(feasible_prefixes) > 0:
                        avg_sim_feas = scipy.mean(list(map(lambda f: f.similarity(p, self.args.gamma), feasible_prefixes)))
                    if len(infeasible_prefixes) > 0:
                        avg_sim_infeas = scipy.mean(list(map(lambda f: f.similarity(p, self.args.gamma), infeasible_prefixes)))

                    obj = avg_sim_feas - avg_sim_infeas

                    logging.debug(plan_fname + " Avg. Similarity to Feasible Prefixes: " + str(avg_sim_feas) + " Avg. Similarity to Infeasible Prefixes: " + str(avg_sim_infeas) + " Objective: " + str(obj))

                    if obj > best_obj:
                        best_plans = [plan_fname]
                        best_obj = obj
                    elif obj == best_obj:
                        best_plans.append(plan_fname)
        return random.choice(best_plans)


    def solveLGPDiverse(self):
        """Main function for LGP Diverse Solver"""
        self.start_time = time.time()

        logging.info("Generating PDDL files")        
        ret = self.callLGPGen()

        # write to file for debugging
        out = ret.stdout.decode("utf-8")
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
        filename = self.args.debugDir + "/" + "z.lgpgen" + dt_string 
        with open(filename, 'w') as f:
            f.write(out)

        self.solved = False
        while not self.solved:
            logging.info("Generating plans")
            self.generateDiversePlans(self.args.N)
            logging.info("Plans generated")

            logging.info("Choosing next plan to test")
            chosen_plan = self.chooseNextPlan()
            logging.info("Chosen plan: " +  chosen_plan)

            logging.info("Testing feasibility")
            chosen_plan_path = os.path.join(self.args.conflictsDir, chosen_plan)

            p = Plan("")
            p.read_from_file(chosen_plan_path)

            # check if it is OK
            # I want to avoid action sequences of type '(place block2 r_gripper block2)'.
            # continue here!

            # NOTE: checkLoop1 reduces the number of plans
            # 1       eager   1.0     0       True    65      58      41      41      34      102.16839480400085
            # 1       eager   1.0     0       True    109     109     41      41      33      113.95562863349915
            index = p.checkLoop1() 
            if index ==  -1:
                ret, feasible, conflict = self.callLGPCheck(chosen_plan_path)
            else:
                logging.info("Skipping plan because contains loop of length 1")        
                conflict = p[0:index+1]
                feasible = False

            if conflict is None: # Could happen if logical plan is invalid in LGP
                conflict = self.plans[chosen_plan]
            logging.info("Done testing feasibility " + str(feasible))        
            if feasible == 1:                
                logging.info("Solved: " + chosen_plan)
                self.solution = self.plans[chosen_plan]
                self.solved = True
            else:
                self.tested_plans[chosen_plan] = conflict
                shutil.copy( os.path.join(self.args.conflictsDir, chosen_plan), os.path.join(self.args.conflictsDir, "original"))
                conflict.write_to_file(os.path.join(self.args.conflictsDir, chosen_plan))     
                # Find all existing plans eliminated by this conflict
                
                for plan_fname, plan in self.plans.items():
                    if plan.actions[:len(conflict)] == conflict.actions:
                        logging.debug("Plan " + plan_fname + " eliminated by new conflict")
                        self.eliminated_plans[plan_fname] = chosen_plan
                

                
            if self.args.timeout > 0 and time.time() - self.start_time > self.args.timeout:
                logging.warning("timeout")
                break

        self.end_time = time.time()   

    def reportStatistics(self):
        #config, N, conflict, gamma, novelty, seed, solved, plan length, plans generated, plans tested, additional feasibility checks pose, additional feasibility checks seq, additional feasibility checks path, total time
        print(self.args.config, self.args.N, self.args.conflict, self.args.gamma, self.args.novelty, self.args.seed, 
        self.solved, len(self.solution), 
        self.num_generated_plans, self.num_tested_plans, 
        self.num_feasibility_checks_pose, self.num_feasibility_checks_seq, self.num_feasibility_checks_path,         
        self.end_time - self.start_time, sep="\t")


            





def main():
    """main function"""
    parser = argparse.ArgumentParser(description='Use diverse PDDL planning to solve LGP problems', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--display', dest='display', help='display feasible plan', action='store_true')
    parser.add_argument('--config', type=str, help='path of rai.cfg', default="rai.cfg")

    
    parser.add_argument('--N', type=int, help='number of plans to generate per call to diverse planner', default=1)  
    parser.add_argument('--conflict', type=str, choices=["lazy","eager","mr"], help="conflict extraction mode", default="eager")
    parser.add_argument('--gamma', type=float, help="discount factor for plan similarity", default=1.0)
    parser.add_argument('--novelty', dest='novelty', help='use novelty to choose next plan', action='store_true')

    parser.add_argument('--timeout', type=float, help='timeout for single planning call (in seconds), or 0 for no timeout', default=0.0)

    parser.add_argument('--seed', type=int, help='random number generator seed', default=0)

    parser.add_argument('--conflictsDir', type=str, help='location of plans', default="found_plans/conflicts")
    parser.add_argument('--debugDir', type=str, help='location of plans', default="found_plans/debug")
    parser.add_argument('--basePlanFilename', type=str, help='base filename for plans', default="sas_plan")
    parser.add_argument('--cache', type=str, help='name of file that contains the cache', default="cache.json")

    parser.add_argument('--LGPExecutable', type=str, help='location of LGP executable', default="./x.exe")
    parser.add_argument('--ForbidIterativeExecutable', type=str, help='location of Forbid Iterative executable', default="../forbiditerative/fast-downward.py")
    parser.add_argument('--ForbidIterativeBuild', type=str, help='name of Forbid Iterative build', default="release64")
    parser.add_argument('--ForbidIterativePlannerAlias', type=str, help='name of FD planner alias to use', default="lama-first")

    args = parser.parse_args()
    
    logging.basicConfig(filename="lgpDiverse.log", level=logging.DEBUG)

    solver = LGPDiverseSolver(args)
    solver.solveLGPDiverse()
    solver.reportStatistics()


if __name__ == '__main__':
    main()    
