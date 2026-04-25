import glob
import os.path
import json

plansdir = "found_plans"

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

def read_data(plansdir):
    feasible_plans = []
    infeasible_plans = []

    with open("cache.json") as json_f:
        cache = json.load(json_f)

    feasible_prefixes = list(map(lambda x: Plan(x), cache['feasible_prefixes']))
    infeasible_prefixes = list(map(lambda x: Plan(x), cache['infeasible_prefixes']))
    
    files = glob.glob(os.path.join(plansdir, "sas_plan.*" ))
    for fname in files:
        if fname[-4:] != ".log" and fname[-1] in ['0', '2', '4', '6', '8']:
            p = Plan("")
            p.read_from_file(fname)
            
            
            total_sim_feas = 0
            for f in feasible_prefixes:
                total_sim_feas = total_sim_feas + similarity(f, p)
            avg_sim_feas = total_sim_feas / len(feasible_prefixes)

            total_sim_infeas = 0
            for f in infeasible_prefixes:
                total_sim_infeas = total_sim_infeas + similarity(f, p)
            avg_sim_infeas = total_sim_feas / len(infeasible_prefixes)

            print(fname, avg_sim_feas, avg_sim_infeas)

    
if __name__ == '__main__':
    read_data(plansdir)