world {}

#thick radius floor and walls

floor (world): { shape: ssBox, Q: [0, 0, .05], size: [2.4, 2.4, .1, .04], color: [.7] contact: 1 friction: 10, logical:{ is_place }}

wall_right (floor): { shape: ssBox, Q: [0, -1.1, 0.2], size: [2.4, .2, .3, .04], color: [.3], contact: 1 }
wall_back (floor): { shape: ssBox, Q: [1.1, 0, 0.2], size: [.2, 2.4, .3, .04], color: [.3], contact: 1 }
wall_left (floor): { shape: ssBox, Q: [0, 1.1, 0.2], size: [2.4, .2, .3, .04], color: [.3], contact: 1 }
wall_front (floor): { shape: ssBox, Q: [-1.1, 0, 0.2], size: [.2, 2.4, .3, .04] , color: [.3], contact: 1 }



### camera
        
camera_gl(world): { Q: "t(0 0 12) d(180 0 0 1)" shape: camera width: 400 height: 400}

block1(floor): { shape: ssBox, Q: [.7, .7, 0.15], size: [.6, .6, .2, .04], color: [.5], contact: 1 }

block2(floor): { shape: ssBox, Q: [.7, -.5, 0.15], size: [.6, 1., .2, .04], color: [.5], contact: 1 }
        
block3(floor): { shape: ssBox, Q: [-.1, -.1, 0.15], size: [.2, .2, .2, .04], color: [.5], contact: 1 }

block4(floor): { shape: ssBox, Q: [-.5, .7, 0.15], size: [1., .6, .2, .04], color: [.5], contact: 1 }
        
base(world): { Q: [0, 0, .2] }
ego(base): { joint: transXY, limits: [-.8, -.8, .8, .8], sampleUniform: 1., shape: ssCylinder, Q: [.2, .2, 0],  size: [.2, .2, .04], color: [0, 1, 1], contact: 1, logical: { is_gripper } }
obj(floor): { joint: rigid, shape: ssCylinder, Q: [.7, .2, .15],  size: [.2, .2, .04], color: [1., .5, .0], contact: 1, logical:{ is_object } }
goal(floor): { shape: ssBox, Q: [.2, .7, 0],  size: [.4, .4, .1, .04], color: [.5, 1., 0.], logical: { is_place } }

camera_init: { X: [0 0 3 0 1 0 0], width: 400 height: 400, focalLength: 1, zRange: [.5, 10]}
