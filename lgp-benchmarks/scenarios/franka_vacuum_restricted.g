Include: <../rai-robotModels/scenarios/panda_noGripper.g>

# filling the hole
gripper_fill (panda_joint8){ shape:cylinder, color:[.1, .1, .1 ,1], size:[.17 .031]}

gripperBall (panda_joint7){
    shape:sphere, size:[.02]
    Q: "t(0 0 .2)"
    contact:0 
    color:[1.0,0.5,0.5,1.0]
}

#gripper (panda_joint7){
#    shape:sphere, size:[.02]
#    Q: "t(.125 0.0 .0)"
#    contact:0 
#    color:[1.0,0.5,0.5,1.0]
#}

gripper_p1 (panda_joint7){
    shape:sphere, size:[.02]
    Q: "t(0.01 -0.03 0.2)"
    contact:0
    color:[1.0,0.5,0.5,1.0]
}

gripper_p2 (panda_joint7){
    shape:sphere, size:[.02]
    Q: "t(0.01 0.03 0.2)"
    contact:0
    color:[1.0,0.5,0.5,1.0]
}

gripper_p3 (panda_joint7){
    shape:sphere, size:[.02]
    Q: "t(-0.032 -0.0 0.2)"
    contact:0
    color:[1.0,0.5,0.5,1.0]
}

gripper_vis (panda_joint7){
    shape:ssCylinder, size:[.04 0.05 .01]
    Q: "t(0.0 0 0.2) d(0 0 1 0)"
    contact:0
    color:[.8,0.8,0.5,.5]
}

gripper_coll (panda_joint7){
    shape:ssCylinder, size:[.03 0.05 .01]
    Q: "t(0. 0.0 0.2) d(0 0 1 0)"
    contact:0
    color:[1.0,0.5,0.5,.5]
}
