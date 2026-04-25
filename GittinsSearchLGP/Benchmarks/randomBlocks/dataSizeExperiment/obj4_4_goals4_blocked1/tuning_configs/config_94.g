world: { pose: [0, 0, 0.1] }
floor(world): { pose: [0, 0, -0.05], shape: ssBox, size: [10, 10, 0.1, 0.04], color: [0.5, 0.1], friction: 10, logical: { is_place: True } }
floor_visual(floor): { shape: ssBox, size: [10, 10, 0.08, 0.04], color: [0.6953, 0.515625, 0.453125] }
wall_right(floor): { pose: [0, -5, 0.2], shape: ssBox, size: [10, 0.2, 0.3, 0.04], color: [0.6953, 0.515625, 0.453125], contact: 1 }
wall_back(floor): { pose: [5, 0, 0.2], shape: ssBox, size: [0.2, 10, 0.3, 0.04], color: [0.6953, 0.515625, 0.453125], contact: 1 }
wall_left(floor): { pose: [0, 5, 0.2], shape: ssBox, size: [10, 0.2, 0.3, 0.04], color: [0.6953, 0.515625, 0.453125], contact: 1 }
wall_front(floor): { pose: [-5, 0, 0.2], shape: ssBox, size: [0.2, 10, 0.3, 0.04], color: [0.6953, 0.515625, 0.453125], contact: 1 }
base(world): { pose: [0, 0, 0.1] }
ego(base): { joint: transXY, limits: [-5, -5, 5, 5], shape: ssCylinder, size: [0.2, 0.2, 0.02], color: [0.96875, 0.742188, 0.308594], contact: 1, logical: { is_gripper: True }, sampleUniform: 1 }
camera_init: { pose: [0, 0, 11, 0, 1, 0, 0], width: 1000, height: 1000, focalLength: 1, zRange: [0.5, 100] }
objectA(floor): { pose: [-0.105089, -2.93134, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjA_north(floor): { pose: [-0.105089, -2.48134, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjA_south(floor): { pose: [-0.105089, -3.38134, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
objectB(floor): { pose: [2.55853, -2.7149, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjB_south(floor): { pose: [2.55853, -3.1649, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjB_west(floor): { pose: [2.10853, -2.7149, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
objectC(floor): { pose: [-3.40327, 0.932533, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjC_north(floor): { pose: [-3.40327, 1.38253, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjC_west(floor): { pose: [-3.85327, 0.932533, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
objectD(floor): { pose: [-0.745766, 2.74583, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjD_east(floor): { pose: [-0.295766, 2.74583, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjD_south(floor): { pose: [-0.745766, 2.29583, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalA(floor): { pose: [-2.50312, -2.09084, 0], shape: ssBox, size: [0.5, 0.5, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True } }
wallgoalA_south(floor): { pose: [-2.50312, -2.39084, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalB(floor): { pose: [-3.48025, 3.28105, 0], shape: ssBox, size: [0.5, 0.5, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True } }
wallgoalB_west(floor): { pose: [-3.78025, 3.28105, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalC(floor): { pose: [2.71306, 2.40615, 0], shape: ssBox, size: [0.5, 0.5, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True } }
wallgoalC_east(floor): { pose: [3.01306, 2.40615, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalD(floor): { pose: [2.62554, 0.388675, 0], shape: ssBox, size: [0.5, 0.5, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True } }
wallgoalD_south(floor): { pose: [2.62554, 0.088675, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }