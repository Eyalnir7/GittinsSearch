import sys
import os


# Read Plan to Check
plan_to_check_filename = sys.argv[1]

with open(plan_to_check_filename,"r") as f:
    lines = f.readlines()
    f.close()

plan_to_check = []
for line in lines:
    sline = line.strip()
    if sline[0] != ";":
        plan_to_check.append(sline)


# Read conflicts from previous plans
min_conflicts_filename = sys.argv[2]

with open(min_conflicts_filename,"r") as f:
    clines = f.readlines()
    f.close()

conflicts = []
for cline in clines:
    actions = map(lambda act: "(" + act.strip(), cline.strip().split("(")[1:])
    conflicts.append(actions)
    

# Check if any conflict is a prefix of the plan to check
conflict_found = False
for conflict in conflicts:
    if len(plan_to_check) >= len(conflict) and plan_to_check[:len(conflict)] == conflict:
        print "Conflict: found", conflict
        conflict_found = True
        break

if not conflict_found:
    print "Conflict: not found"