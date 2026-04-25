Include: <_base-walls.g>

base(world): { Q: [0, 0, .1] }

ego(base) {
    shape: ssCylinder, size: [.2, .2, .02], color: [0.96875, 0.7421875, 0.30859375], logical: {is_gripper}, limits: [-1.8, -1.8, 1.8, 1.8], sampleUniform: 1,
    joint: transXY, contact: 1
}

goal (floor): { shape: ssBox, Q: "t(-1.7 1.7 0.0)", size: [0.2, 0.2, .1, .04], color: [1., .3, .3, 0.9], contact: 0, logical: { is_place } }

wall1_v (floor): { shape: ssBox, Q: "t(-1.0 1.5 0.2)", size: [.1, 1, .3, .02],color: [0.6953, 0.515625, .453125], contact: 1 }
wall2_h (floor): { shape: ssBox, Q: "t(-0.5 1.  0.2)", size: [1, .1, .3, .02], color: [0.6953, 0.515625, .453125],contact: 1 }
wall3_v (floor): { shape: ssBox, Q: "t(-1 0.05 0.2)", size: [.1, 1, .3, .02], color: [0.6953, 0.515625, .453125], contact: 1 }
wall4_h (floor): { shape: ssBox, Q: "t(-0.5 0.5 0.2)", size: [1, .1, .3, .02], color: [0.6953, 0.515625, .453125], contact: 1 }

wall5_v (floor): { shape: ssBox, Q: "t(-1.55 0.05 0.2)", size: [0.1, 1, .3, .02], color: [0.6953, 0.515625, .453125], contact: 1 }
wall6_h (floor): { shape: ssBox, Q: "t(-1.8 0.5 0.2)", size: [0.4, .1, .3, .02], color: [0.6953, 0.515625, .453125], contact: 1 }
wall8_h (floor): { shape: ssBox, Q: "t(-1.75 1.0 0.2)", size: [0.5, .1, .3, .02], color: [0.6953, 0.515625, .453125], contact: 1 }

obj(floor) {
    shape: ssBox, Q: [-0.5, 1.4, .2], size: [.3, .3, .3, .02], nomass: 1,  color: [0.0, 0.0, 1.0],
    joint: rigid, friction: .1 contact: 1
}

camera_init: { X: [0 0 5 0 1 0 0], width: 400 height: 400, focalLength: 1, zRange: [.5, 10]}
