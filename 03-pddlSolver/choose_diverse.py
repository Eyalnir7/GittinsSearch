import glob
import os.path
import argparse
import json
import scipy
import math

class Plan:
    def __init__(self, plan_string):
        self.plan_string = plan_string
        self.process_plan_string()

    def process_plan_string(self):
        ind = self.plan_string.find(";")
        if ind >= 0:
            self.plan_string = self.plan_string[:ind]

        self.actions = list(map(
            lambda a: a.strip() + ")", 
                filter(lambda a: len(a.strip()) > 0,
                        self.plan_string.split(")"))))        

    def read_from_file(self, filename):
        with open(filename) as f:
            self.plan_string = f.read()
        self.process_plan_string()

    def __repr__(self):
        return "|".join(self.actions)


def similarity(plan1, plan2):
    actions1 = set(plan1.actions)
    actions2 = set(plan2.actions)
    intersection = len(list(set(actions1).intersection(actions2)))
    union = (len(actions1) + len(actions2)) - intersection
    return float(intersection) / union



def find_best_plan(args):
    with open(args.cache) as json_f:
        cache = json.load(json_f)

    feasible_prefixes = list(map(lambda x: Plan(x), cache['feasible_prefixes']))
    infeasible_prefixes = list(map(lambda x: Plan(x), cache['infeasible_prefixes']))
    
    files = glob.glob(os.path.join(args.plansdir, args.base_plan_filename + ".[0-9]*" ))    
    best_obj = -math.inf
    best_indices = []
    for fname in files:
        ind = fname.rfind(".")
        suffix = fname[ind+1:]
        if suffix.isdigit():
            num = int(fname[ind+1:])-1
            if num > args.num_plans_to_skip:
                p = Plan("")
                p.read_from_file(fname)

                avg_sim_feas = 0.0
                avg_sim_infeas = 0.0

                if len(feasible_prefixes) > 0:
                    avg_sim_feas = scipy.mean(list(map(lambda f: similarity(f, p), feasible_prefixes)))
                if len(infeasible_prefixes) > 0:
                    avg_sim_infeas = scipy.mean(list(map(lambda f: similarity(f, p), infeasible_prefixes)))

                obj = avg_sim_feas - avg_sim_infeas

                print(fname, "Avg. Similarity to Feasible Prefixes: ", avg_sim_feas,  "Avg. Similarity to Infeasible Prefixes: ", avg_sim_infeas, "Objective: ", obj, sep="\t")

                if obj > best_obj:
                    best_indices = [num]
                    best_obj = obj
                elif obj == best_obj:
                    best_indices.append(num)

    print('Plan index: ', best_indices[0])
    

    
def main():
    parser = argparse.ArgumentParser(description='Choose a plan which maximizes similarity to feasible prefixes and minimizes similarity to infeasible prefixes')

    parser.add_argument('--plansdir', type=str, help='directory where plans will be found', default="found_plans")
    parser.add_argument('--base-plan-filename', type=str, help='base filename for plans', default="sas_plan")
    parser.add_argument('--cache', type=str, help='name of file that contains the cache', default="cache.json")
    parser.add_argument('--num-plans-to-skip', type=int, help='index of last plan to skip', default=1)

    args = parser.parse_args()
    find_best_plan(args)

if __name__ == '__main__':
    main()    