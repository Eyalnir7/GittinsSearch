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
objectA(floor): { pose: [-0.384669, 2.02618, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjA_north(floor): { pose: [-0.384669, 2.47618, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjA_south(floor): { pose: [-0.384669, 1.57618, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
objectB(floor): { pose: [0.509706, -2.87224, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjB_north(floor): { pose: [0.509706, -2.42224, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjB_east(floor): { pose: [0.959706, -2.87224, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
objectC(floor): { pose: [-3.36542, -1.98783, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjC_south(floor): { pose: [-3.36542, -2.43783, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjC_north(floor): { pose: [-3.36542, -1.53783, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
objectD(floor): { pose: [2.76392, 0.822463, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjD_south(floor): { pose: [2.76392, 0.372463, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjD_west(floor): { pose: [2.31392, 0.822463, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalA(floor): { pose: [-3.39475, 2.38625, 0], shape: ssBox, size: [0.5, 0.5, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True } }
wallgoalA_east(floor): { pose: [-3.09475, 2.38625, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalB(floor): { pose: [1.58616, 3.5344, 0], shape: ssBox, size: [0.5, 0.5, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True } }
wallgoalB_south(floor): { pose: [1.58616, 3.2344, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalC(floor): { pose: [2.56646, -1.2058, 0], shape: ssBox, size: [0.5, 0.5, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True } }
wallgoalC_north(floor): { pose: [2.56646, -0.905796, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalD(floor): { pose: [2.5741, -3.63692, 0], shape: ssBox, size: [0.32, 0.32, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True, busy: True } }
wallobjD(goalD): { pose: [0, 0, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0.7, 0.7, 0.7], contact: 1, logical: { is_object: True, is_obstacle: True, on_goal: True, busy: True, on: True } }