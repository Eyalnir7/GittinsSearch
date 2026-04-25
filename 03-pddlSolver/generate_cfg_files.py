import sys
from pathlib import Path
timeout = sys.argv[1]

# python get current working directory
base_path = Path.cwd().parent.as_posix()
folFile = base_path + "/models/fol-pnp-push.g"

confFiles = [
    base_path+"/models/workshopTable-push-tiny.g", # ball, stick
    base_path+"/models/workshopTable-push-small.g", # ball, stick, block1
    base_path+"/models/workshopTable-push.g", # ball, stick,  block1 , block2
]

goals_tiny = [
"(on r_gripper ball2)", # Requires 4 actions. TINY OK ;  SMALL OK ; STANDARD OK
"(on l_gripper ball2)" # Requires 5 actions. TINY OK ; SMALL OK (longer) ;  Timeout 100 seconds 
]

goals_small = [
"(on block1 ball2)", # Requires 5 actions; SMALL OK ; Standard OK
"(on block1 ball2) (on r_gripper block1)" # Requires 6 actions; SMALL OK ; either timeout 100 second / solve around 100 seconds
]

goals_standard = [
"(on block1 ball2) (on block2 block1)" # Requires 7 actions; Standard; timeout 100 seconds
]

goals = [goals_tiny, goals_small, goals_standard]

names = ["tiny", "small", "standard"]

for i, gs in enumerate(goals):
    for j, g in enumerate(gs):
        for k, confFile in enumerate(confFiles[i:]):
            filename = "rai-push-c{}-g{}-{}.cfg".format(names[i+k],names[i],j)
            with open(filename, "w") as f:
                f.write('folFile: "{}"\n'.format(folFile))
                f.write('confFile: "{}"\n'.format(confFile))
                f.write('planFile: "kaka.sas_plan"\n')
                f.write('sktFile: "tmp.skt"\n')
                f.write('goal: "{}"\n'.format(g))
                f.write('LGP/verbose:2\n')
                f.write('LGP/stopSol:1\n')
                f.write('LGP/stopTime:'  + timeout + '\n')
            print(filename)



folFile = base_path + "/models/fol-pnp.g"
confFile = base_path + "/models/workshopTable-small.g"
goals = ["(on board1 block1)",
         "(on box1 block1) (on block1 block2)",
         "(on box1 block1) (on block1 block2) (on block2 block3)",
         "(on box1 block1) (on block1 block2) (on block2 block3) (on block3 block4)",
         "(on box1 block1) (on block1 block2) (on block2 block3) (on block3 block4) (on block4 block5)",
         "(on box2 block1) (on block1 block2) (on block2 block3) (on block3 block4) (on block4 block5)"]
for i,g in enumerate(goals):
    filename = "rai-blocks-{}.cfg".format(i)
    with open(filename, "w") as f:
        f.write('folFile: "{}"\n'.format(folFile))
        f.write('confFile: "{}"\n'.format(confFile))
        f.write('planFile: "kaka.sas_plan"\n')
        f.write('sktFile: "tmp.skt"\n')
        f.write('goal: "{}"\n'.format(g))
        f.write('LGP/verbose:2\n')
        f.write('LGP/stopSol:1\n')
        f.write('LGP/stopTime:'  + timeout + '\n')
    print(filename)       


folFile = base_path + "/models/fol-pnp-hanoi.g"
confFile = base_path +  "/models/workshop-hanoi.g"
goals = ["(on box2 block3)",
#"(on box2 block2) (on block2 block3)" # This does not work because you can not do handover...
"(on box3 block2) (on block2 block3)",
"(on box3 block1) (on block1 block2) (on block2 block3)"]

for i,g in enumerate(goals):
    filename = "rai-hanoi-{}.cfg".format(i)
    with open(filename, "w") as f:
        f.write('folFile: "{}"\n'.format(folFile))
        f.write('confFile: "{}"\n'.format(confFile))
        f.write('planFile: "kaka.sas_plan"\n')
        f.write('sktFile: "tmp.skt"\n')
        f.write('goal: "{}"\n'.format(g))
        f.write('LGP/verbose:2\n')
        f.write('LGP/stopSol:1\n')
        f.write('LGP/stopTime:'  + timeout + '\n')
    print(filename)       

