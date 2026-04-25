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
objectA(floor): { pose: [3.57134, -0.36464, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjA_south(floor): { pose: [3.57134, -0.81464, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjA_north(floor): { pose: [3.57134, 0.0853604, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
objectB(floor): { pose: [-1.06317, 1.79279, 0.2], joint: rigid, shape: ssBox, size: [0.3, 0.3, 0.3, 0.02], color: [0, 0, 1], contact: 1, logical: { is_object: True } }
wallobjB_north(floor): { pose: [-1.06317, 2.24279, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
wallobjB_east(floor): { pose: [-0.613167, 1.79279, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalA(floor): { pose: [2.95214, 3.43469, 0], shape: ssBox, size: [0.5, 0.5, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True } }
wallgoalA_west(floor): { pose: [2.65214, 3.43469, 0.15], shape: ssBox, size: [0.15, 0.8, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }
goalB(floor): { pose: [-3.05931, 2.74855, 0], shape: ssBox, size: [0.5, 0.5, 0.1, 0.01], color: [1, 0.3, 0.3], logical: { is_goal: True, is_place: True } }
wallgoalB_south(floor): { pose: [-3.05931, 2.44855, 0.15], shape: ssBox, size: [0.8, 0.15, 0.3, 0.01], color: [0.7, 0.7, 0.7], contact: 1 }