frame endeffA {
    X:<t(0 0 .1)>, ctrl_H,
    shape:sphere size:[.1] color:[1. .5 0], contact:-1,
    logical:{gripper, support} }

frame arm0(endeffA) {
    joint:hingeZ A:<t(0 0 .11)>
    shape:ssBox mass:1 size:[0.1 0.1 .25 .03], contact:-1 }

frame arm1(arm0) {
    joint:hingeX A:<t(0 0 .11)> B:<t(0 0 .11)> q:.5
    shape:ssBox mass:1 size:[0.1 0.1 .25 .03], contact:-1 }

frame mid1(arm1) {
    joint:hingeX A:<t(0 0 .11)> B:<t(0 0 .22)> q:.5 
    shape:ssBox mass:1 size:[0.1 0.1 .5 .03], contact:-1 }

frame mid2(mid1) {
    joint:hingeX A:<t(0 0 .22)> B:<t(0 0 .11)> q:.5 
    shape:ssBox mass:1 size:[0.1 0.1 .25 .03], contact:-1 }

frame arm2(mid2) {
    joint:hingeX A:<t(0 0 .11)> B:<t(0 0 .11)> q:.5
    shape:ssBox mass:1 size:[0.1 0.1 .25 .03], contact:-1 }

frame endeffB(arm2) {
    joint:hingeZ A:<t(0 0 .11)>, ctrl_H,
    shape:sphere size:[.1] color:[1. 0 .5], contact:-1,
    logical:{gripper} }


