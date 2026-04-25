Include: <_base-walls.g>

base(world): { Q: [0, 0, .1] }

ego(base) {
    shape: ssCylinder, size: [.2, .2, .02], color: [0.96875, 0.7421875, 0.30859375], logical: {is_gripper}, limits: [-5, -5, 5, 5], sampleUniform: 1,
    joint: transXY, contact: 1
}

#goal1 (floor): { shape: ssBox, Q: [-1.5 1.5 .0], size: [1.0, 1.0, .1, .#02], color: [1., .3, .3], contact: 0, logical: { is_place, is_goal } }
#goal2 (floor): { shape: ssBox, Q: [2.5 2.5 .0], size: [1.0, 1.0, .1, .02], #color: [1., .3, .3], contact: 0, logical: { is_place, is_goal } }

#obj1(floor) {
#    shape: ssBox, Q: [1.5, -0.5, .2], size: [.3, .3, 0.3, .02], nomass: 1, color: [0, 0, 1.0],
#    joint: rigid, friction: .1 contact: 1, logical : {is_object}
#}



camera_init: { X: [0 0 11 0 1 0 0], width: 1000 height: 1000, focalLength: 1, zRange: [.5, 100]}
