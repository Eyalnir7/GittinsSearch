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
objectA(floor): { pose: [1.7178, -1.41163, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjA_east(floor): { pose: [2.1678, -1.41163, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjA_north(floor): { pose: [1.7178, -0.961626, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
objectB(floor): { pose: [3.57485, -0.64665, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjB_south(floor): { pose: [3.57485, -1.09665, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjB_east(floor): { pose: [4.02485, -0.64665, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
objectC(floor): { pose: [-0.630118, -2.73052, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjC_south(floor): { pose: [-0.630118, -3.18052, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjC_north(floor): { pose: [-0.630118, -2.28052, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
objectD(floor): { pose: [-0.300259, 3.5129, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjD_west(floor): { pose: [-0.750259, 3.5129, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjD_south(floor): { pose: [-0.300259, 3.0629, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalA(floor): { pose: [1.8002, 1.00708, 0], shape: ssBox, size: [0.5, 0.5, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True } }
wallgoalA_east(floor): { pose: [2.1002, 1.00708, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalB(floor): { pose: [-2.5814, 0.828688, 0], shape: ssBox, size: [0.5, 0.5, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True } }
wallgoalB_east(floor): { pose: [-2.2814, 0.828688, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalC(floor): { pose: [-3.62301, -3.55538, 0], shape: ssBox, size: [0.5, 0.5, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True } }
wallgoalC_north(floor): { pose: [-3.62301, -3.25538, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalD(floor): { pose: [2.4784, 2.98193, 0], shape: ssBox, size: [0.5, 0.5, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True } }
wallgoalD_north(floor): { pose: [2.4784, 3.28193, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }