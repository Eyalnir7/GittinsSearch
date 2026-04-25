Include: <../../rai-robotModels/scenarios/panda_fixRobotiq.g>

gripperBall (panda_joint7){
    shape:sphere, size:[.02]
    Q:"t(0 0 .27)"
    contact:0 
    color:[1.0,0.5,0.5,1.0],
    logical: { is_gripper }
}
