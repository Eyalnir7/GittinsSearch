Include: <_base-walls.g>

base(world): { Q: [0, 0, .1] }

ego(base) {
    shape: ssCylinder, size: [.2, .2, .02], color: [0.96875, 0.7421875, 0.30859375], logical: {is_gripper}, limits: [-1.8, -1.8, 1.8, 1.8], sampleUniform: 1,
    joint: transXY, contact: 1
}

goal (floor): { shape: ssBox, Q: [-1.5 1.5 .0], size: [1.0, 1.0, .1, .02], color: [1., .3, .3], contact: 0, logical: { is_place } }

block1(floor) {
    shape: ssBox, Q: [-1.5, 1, .15], size: [0.8, 0.2, .2, .02], nomass: 1,  color: [1.0, 1.0, 1.0],
    joint: rigid, friction: .1 contact: 1
}

block2(floor) {
    shape: ssBox, Q: [-0.9, 1.4, .15], size: [.2, 1, .2, .02], nomass: 1,  color: [1.0, 1.0, 1.0],
    joint: rigid, friction: .1 contact: 1
}

obj(floor) {
    shape: ssBox, Q: [1.5, -0.5, .2], size: [.3, .3, 0.3, .02], nomass: 1, color: [0, 0, 1.0],
    joint: rigid, friction: .1 contact: 1
}

camera_init: { X: [0 0 5 0 1 0 0], width: 400 height: 400, focalLength: 1, zRange: [.5, 10]}
