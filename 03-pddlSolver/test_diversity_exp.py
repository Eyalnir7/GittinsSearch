import scipy
import re
import glob
import os.path

lpfile = "test_instance.lp"
plansdir = "found_plans"


def read_data(lpfile, plansdir):
    similarity = {}
    feasible_plans = []
    infeasible_plans = []

    pattern = re.compile("Score for pair (?P<x1>[0-9]+), (?P<x2>[0-9]+): (?P<x3>\-?[0-9]+(\.[0-9]+)?)")
    
    with open("scores.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            m = pattern.match(line)
            if m is not None:
                p1 = int(m.group(1))+1
                p2 = int(m.group(2))+1
                s = float(m.group(3))
                similarity[(p1, p2)] = s
                similarity[(p2, p1)] = s

    f_pattern = re.compile("Final verdict - feasible: (?P<f>[0-1])")
    n_pattern = re.compile(os.path.join(plansdir, "sas_plan.(?P<n>[0-9]+).komo.log"))
    files = glob.glob(os.path.join(plansdir, "sas_plan.*.komo.log" ))
    for fname in files:
        with open(fname, "r") as f:
            nm = n_pattern.match(fname)
            n = int(nm.group(1))

            lines = f.readlines()
            last_line = lines[-1]
            fm = f_pattern.match(last_line)
            fs = 0
            if fm is not None:
                fs = int(fm.group(1))
            if fs == 0:
                infeasible_plans.append(n)
            elif fs == 1:
                feasible_plans.append(n)

    
    ffs = []
    for i,x in enumerate(feasible_plans):
        for y in feasible_plans[i+1:]:
            ffs.append(similarity[(x,y)])

    iis = []
    for i,x in enumerate(infeasible_plans):
        for y in infeasible_plans[i+1:]:
            iis.append(similarity[(x,y)])

    
    fis = []
    for x in feasible_plans:
        for y in infeasible_plans:
            fis.append(similarity[(x,y)])

    print("Feasible plan similarities", scipy.mean(ffs), ffs)
    print("Infeasible plan similarities", scipy.mean(fis), fis)
    print("Feasible/infeasible plan similarities", scipy.mean(fis), fis)

    
if __name__ == '__main__':
    read_data(lpfile, plansdir)