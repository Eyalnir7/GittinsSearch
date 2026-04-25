world: {  }
table1: { pose: [0, 2, 0.6, 0.707107, 0, 0, 0.707107], shape: ssBox, size: [0.8, 2.2, 0.1, 0.02], color: [0.3, 0.3, 0.3], contact: 1, fixed: True, logical: {  } }
table2: { pose: [2, 0, 0.6], shape: ssBox, size: [0.8, 2.2, 0.1, 0.02], color: [0.3, 0.3, 0.3], contact: 1, fixed: True, logical: { } }
table3: { pose: [0, -2, 0.6, 0.707107, 0, 0, 0.707107], shape: ssBox, size: [0.8, 2.2, 0.1, 0.02], color: [0.3, 0.3, 0.3], contact: 1, fixed: True, logical: {  } }

worldTranslationRotation(world): { pose: [0.707107, 0, 0, 0.707107], joint: transXYPhi, limits: [-10, -10, -6, 10, 10, 6], shape: marker, size: [0.5, 0, 0, 0], color: [1, 0, 0], gains: [1, 1], base: True }
obj0(table1): { pose: [0.104468, -0.866593, 0.15, 0.940649, 0, 0, -0.339381], joint: rigid, shape: ssBox, size: [0.1, 0.1, 0.2, 0.02], color: [1, 0, 0], contact: 1, logical: { is_object } }
obj1(table1): { pose: [-0.200761, 0.981609, 0.15, 0.395304, 0, 0, 0.91855], joint: rigid, shape: ssBox, size: [0.1, 0.1, 0.2, 0.02], color: [1, 0, 0], contact: 1, logical: { is_object } }
obj2(table1): { pose: [0.103592, 0.175412, 0.15, 0.718158, 0, 0, -0.69588], joint: rigid, shape: ssBox, size: [0.1, 0.1, 0.2, 0.02], color: [1, 0, 0], contact: 1, logical: { is_object } }
obj3(table1): { pose: [0.0558273, -0.45656, 0.15, 0.75111, 0, 0, -0.660177], joint: rigid, shape: ssBox, size: [0.1, 0.1, 0.2, 0.02], color: [1, 0, 0], contact: 1, logical: { is_object } }
tray(table2): { pose: [0, 0, 0.07], shape: ssBox, size: [0.15, 0.15, 0.04, 0.02], color: [0, 1, 0], logical: { is_place } }
base_footprint(worldTranslationRotation): { pose: [0, 0, 0.05], shape: box, size: [0.01, 0.01, 0.01], mass: 1, inertia: [0.01, 0.01, 0.01] }
coll_base(worldTranslationRotation): { pose: [0, 0, 0.17], shape: ssBox, size: [0.7, 0.7, 0.36, 0.1], color: [1, 1, 1, 0.2], contact: -2, coll_pr2: True }
coll_torso(worldTranslationRotation): { pose: [-0.13, 0, 0.65], shape: ssBox, size: [0.45, 0.7, 1.1, 0.1], color: [1, 1, 1, 0.2], contact: -2, coll_pr2: True }
base_footprint_joint(worldTranslationRotation): { pose: [0, 0, 0.101], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/base.h5> }
base_link(worldTranslationRotation): { pose: [0, 0, 0.101], mass: 116, inertia: [5.65223, -0.00971993, 1.29399, 5.66947, -0.00737958, 3.6832] }
base_laser_joint(worldTranslationRotation): { pose: [0.275, 0, 0.353], shape: box, size: [0.001, 0.001, 0.001] }
fl_caster_rotation_joint(worldTranslationRotation): { pose: [0.2246, 0.2246, 0.1292], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/caster.h5> }
fr_caster_rotation_joint(worldTranslationRotation): { pose: [0.2246, -0.2246, 0.1292], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/caster.h5> }
bl_caster_rotation_joint(worldTranslationRotation): { pose: [-0.2246, 0.2246, 0.1292], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/caster.h5> }
br_caster_rotation_joint(worldTranslationRotation): { pose: [-0.2246, -0.2246, 0.1292], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/caster.h5> }
torso_lift_joint(worldTranslationRotation): { pose: [-0.05, 0, 0.940675], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/torso_lift.h5>, gains: [100000, 10], torso: True }
base_laser_link(worldTranslationRotation): { pose: [0.275, 0, 0.353], mass: 0.001, inertia: [0.0001, 1e-06, 0.0001] }
fl_caster_rotation_link(worldTranslationRotation): { pose: [0.2246, 0.2246, 0.1292], mass: 3.47308, inertia: [0.0124118, -0.000711734, 0.00050273, 0.0152182, -4.27347e-06, 0.011764] }
fr_caster_rotation_link(worldTranslationRotation): { pose: [0.2246, -0.2246, 0.1292], mass: 3.47308, inertia: [0.0124118, -0.000711734, 0.00050273, 0.0152182, -4.27347e-06, 0.011764] }
bl_caster_rotation_link(worldTranslationRotation): { pose: [-0.2246, 0.2246, 0.1292], mass: 3.47308, inertia: [0.0124118, -0.000711734, 0.00050273, 0.0152182, -4.27347e-06, 0.011764] }
br_caster_rotation_link(worldTranslationRotation): { pose: [-0.2246, -0.2246, 0.1292], mass: 3.47308, inertia: [0.0124118, -0.000711734, 0.00050273, 0.0152182, -4.27347e-06, 0.011764] }
torso_lift_link(worldTranslationRotation): { pose: [-0.05, 0, 0.940675], mass: 36.248, inertia: [2.77165, 0.00428452, -0.160419, 2.51002, 0.0296645, 0.526432] }
r_shoulder_pan_joint_origin(worldTranslationRotation): { pose: [-0.05, -0.188, 0.940675] }
l_shoulder_pan_joint_origin(worldTranslationRotation): { pose: [-0.05, 0.188, 0.940675] }
endeffBase(worldTranslationRotation): { pose: [-0.05, 0, 0.740675, 0.707107, 0, 0.707107, 0], shape: marker, size: [0.1, 0, 0, 0], color: [1, 0, 0] }
endeffWorkspace(worldTranslationRotation): { pose: [-0.15, 0, 0.340675, 0.5, -0.5, 0.5, -0.5], shape: marker, size: [0.1, 0, 0, 0], color: [1, 0, 0] }
fl_caster_l_wheel_joint(worldTranslationRotation): { pose: [0.2246, 0.2736, 0.1292], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/wheel.h5> }
fl_caster_r_wheel_joint(worldTranslationRotation): { pose: [0.2246, 0.1756, 0.1292], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/wheel.h5> }
fr_caster_l_wheel_joint(worldTranslationRotation): { pose: [0.2246, -0.1756, 0.1292], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/wheel.h5> }
fr_caster_r_wheel_joint(worldTranslationRotation): { pose: [0.2246, -0.2736, 0.1292], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/wheel.h5> }
bl_caster_l_wheel_joint(worldTranslationRotation): { pose: [-0.2246, 0.2736, 0.1292], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/wheel.h5> }
bl_caster_r_wheel_joint(worldTranslationRotation): { pose: [-0.2246, 0.1756, 0.1292], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/wheel.h5> }
br_caster_l_wheel_joint(worldTranslationRotation): { pose: [-0.2246, -0.1756, 0.1292], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/wheel.h5> }
br_caster_r_wheel_joint(worldTranslationRotation): { pose: [-0.2246, -0.2736, 0.1292], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/wheel.h5> }
imu_joint(worldTranslationRotation): { pose: [-0.07977, -0.1497, 1.10467, -1.03412e-13, 0, 1, 0], shape: box, size: [0.001, 0.001, 0.001] }
head_pan_joint(worldTranslationRotation): { pose: [-0.06707, 0, 1.32212], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/head_pan.h5>, gains: [20, 2], head: True }
laser_tilt_mount_joint(worldTranslationRotation): { pose: [0.04893, 0, 1.16768], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/tilting_hokuyo.h5>, laser: True }
fl_caster_l_wheel_link(worldTranslationRotation): { pose: [0.2246, 0.2736, 0.1292], mass: 0.44036, inertia: [0.0124118, -0.000711734, 0.00050273, 0.0152182, -4.27347e-06, 0.011764] }
fl_caster_r_wheel_link(worldTranslationRotation): { pose: [0.2246, 0.1756, 0.1292], mass: 0.44036, inertia: [0.0124118, -0.000711734, 0.00050273, 0.0152182, -4.27347e-06, 0.011764] }
fr_caster_l_wheel_link(worldTranslationRotation): { pose: [0.2246, -0.1756, 0.1292], mass: 0.44036, inertia: [0.0124118, -0.000711734, 0.00050273, 0.0152182, -4.27347e-06, 0.011764] }
fr_caster_r_wheel_link(worldTranslationRotation): { pose: [0.2246, -0.2736, 0.1292], mass: 0.44036, inertia: [0.0124118, -0.000711734, 0.00050273, 0.0152182, -4.27347e-06, 0.011764] }
bl_caster_l_wheel_link(worldTranslationRotation): { pose: [-0.2246, 0.2736, 0.1292], mass: 0.44036, inertia: [0.0124118, -0.000711734, 0.00050273, 0.0152182, -4.27347e-06, 0.011764] }
bl_caster_r_wheel_link(worldTranslationRotation): { pose: [-0.2246, 0.1756, 0.1292], mass: 0.44036, inertia: [0.0124118, -0.000711734, 0.00050273, 0.0152182, -4.27347e-06, 0.011764] }
br_caster_l_wheel_link(worldTranslationRotation): { pose: [-0.2246, -0.1756, 0.1292], mass: 0.44036, inertia: [0.0124118, -0.000711734, 0.00050273, 0.0152182, -4.27347e-06, 0.011764] }
br_caster_r_wheel_link(worldTranslationRotation): { pose: [-0.2246, -0.2736, 0.1292], mass: 0.44036, inertia: [0.0124118, -0.000711734, 0.00050273, 0.0152182, -4.27347e-06, 0.011764] }
imu_link(worldTranslationRotation): { pose: [-0.07977, -0.1497, 1.10467, -1.03412e-13, 0, 1, 0], mass: 0.001, inertia: [0.0001, 1e-06, 0.0001] }
head_pan_link(worldTranslationRotation): { pose: [-0.06707, 0, 1.32212], mass: 1.61112, inertia: [0.00482611, -0.000144684, 0.000110076, 0.00521899, -0.00031424, 0.00861878] }
laser_tilt_mount_link(worldTranslationRotation): { pose: [0.04893, 0, 1.16768], mass: 0.05, inertia: [0.0001, 1e-05, 0.0001] }
endeffLaser(worldTranslationRotation): { pose: [0.0459242, 0.00167121, 1.17989, 0.5, -0.5, 0.5, -0.5], shape: camera, size: [0.1, 0, 0, 0], color: [1, 1, 1], zRange: [0.1, 2] }
head_tilt_joint(worldTranslationRotation): { pose: [0.00093, 0, 1.32212, 0.980067, 0, 0.198669, 0], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/head_tilt.h5>, gains: [60, 4], head: True }
laser_tilt_joint(worldTranslationRotation): { pose: [0.04893, 0, 1.19768], shape: box, size: [0.001, 0.001, 0.001] }
head_tilt_link(worldTranslationRotation): { pose: [0.00093, 0, 1.32212, 0.980067, 0, 0.198669, 0], mass: 1.74973, inertia: [0.0106023, -0.000408814, 0.00198304, 0.0118744, 0.000197909, 0.00551679] }
endeffHead(worldTranslationRotation): { pose: [0.0476602, -0.08, 1.43265, 0.14048, -0.14048, -0.693012, -0.693012], shape: marker, size: [0.1, 0, 0, 0], color: [1, 0, 0] }
endeffEyes(worldTranslationRotation): { pose: [0.0476602, -0.05, 1.43265, 0.14048, 0.14048, -0.693012, 0.693012], shape: marker, size: [0.1, 0, 0, 0], color: [1, 1, 0] }
endeffKinect(worldTranslationRotation): { pose: [0.0985033, 0.147067, 1.59785, 0.390699, -0.589368, 0.589368, -0.390699], shape: camera, size: [0.1, 0, 0, 0], color: [1, 0, 0], zRange: [0.1, 2] }
laser_tilt_link(worldTranslationRotation): { pose: [0.04893, 0, 1.19768], mass: 0.001, inertia: [0.0001, 1e-06, 0.0001] }
head_plate_frame_joint(worldTranslationRotation): { pose: [0.0474161, 0, 1.3725, 0.980067, 0, 0.198669, 0], shape: box, size: [0.01, 0.01, 0.01] }
head_plate_frame(worldTranslationRotation): { pose: [0.0474161, 0, 1.3725, 0.980067, 0, 0.198669, 0], mass: 0.01, inertia: [0.001, 0.001, 0.001] }
sensor_mount_frame_joint(worldTranslationRotation): { pose: [0.0474161, 0, 1.3725, 0.980067, 0, 0.198669, 0], shape: box, size: [0.01, 0.01, 0.01] }
sensor_mount_link(worldTranslationRotation): { pose: [0.0474161, 0, 1.3725, 0.980067, 0, 0.198669, 0], mass: 0.05, inertia: [0.001, 0.001, 0.01] }
high_def_frame(worldTranslationRotation): { pose: [0.0610457, -0.109, 1.40474, 0.980067, 0, 0.198669, 0], mass: 0.01, inertia: [0.001, 0.001, 0.001] }
high_def_frame_0(worldTranslationRotation): { pose: [0.0426245, -0.109, 1.41252, 0.980067, 0, 0.198669, 0], shape: box, size: [0.04, 0.04, 0.04] }
double_stereo_link(worldTranslationRotation): { pose: [0.0485844, 0, 1.37526, 0.980067, 0, 0.198669, 0], mass: 0.1, inertia: [0.001, 0.001, 0.01] }
double_stereo_link_0(worldTranslationRotation): { pose: [0.0491092, 0, 1.40218, 0.980067, 0, 0.198669, 0], shape: box, size: [0.02, 0.12, 0.05] }
wide_stereo_frame_joint(worldTranslationRotation): { pose: [0.0604616, 0.03, 1.40335, 0.980067, 0, 0.198669, 0], shape: box, size: [0.001, 0.001, 0.001] }
narrow_stereo_frame_joint(worldTranslationRotation): { pose: [0.0604616, 0.06, 1.40335, 0.980067, 0, 0.198669, 0], shape: box, size: [0.001, 0.001, 0.001] }
high_def_optical_frame(worldTranslationRotation): { pose: [0.0610457, -0.109, 1.40474, 0.589368, -0.589368, -0.390699, -0.390699], mass: 0.01, inertia: [0.001, 0.001, 0.001] }
high_def_optical_frame_0(worldTranslationRotation): { pose: [0.0610457, -0.109, 1.40474, 0.693012, -0.693012, 0.14048, 0.14048], shape: cylinder, size: [0.05, 0.02] }
wide_stereo_link(worldTranslationRotation): { pose: [0.0604616, 0.03, 1.40335, 0.980067, 0, 0.198669, 0], mass: 0.1, inertia: [0.01, 0.01, 0.01] }
wide_stereo_gazebo_l_stereo_camera_frame_joint(worldTranslationRotation): { pose: [0.0604616, 0.03, 1.40335, 0.980067, 0, 0.198669, 0], shape: box, size: [0.01, 0.01, 0.01] }
narrow_stereo_link(worldTranslationRotation): { pose: [0.0604616, 0.06, 1.40335, 0.980067, 0, 0.198669, 0], mass: 0.1, inertia: [0.01, 0.01, 0.01] }
narrow_stereo_gazebo_l_stereo_camera_frame_joint(worldTranslationRotation): { pose: [0.0604616, 0.06, 1.40335, 0.980067, 0, 0.198669, 0], shape: box, size: [0.01, 0.01, 0.01] }
wide_stereo_gazebo_l_stereo_camera_frame(worldTranslationRotation): { pose: [0.0604616, 0.03, 1.40335, 0.980067, 0, 0.198669, 0], mass: 0.01, inertia: [0.001, 0.001, 0.001] }
narrow_stereo_gazebo_l_stereo_camera_frame(worldTranslationRotation): { pose: [0.0604616, 0.06, 1.40335, 0.980067, 0, 0.198669, 0], mass: 0.01, inertia: [0.001, 0.001, 0.001] }
wide_stereo_gazebo_r_stereo_camera_frame_joint(worldTranslationRotation): { pose: [0.0604616, -0.06, 1.40335, 0.980067, 0, 0.198669, 0], shape: box, size: [0.01, 0.01, 0.01] }
narrow_stereo_gazebo_r_stereo_camera_frame_joint(worldTranslationRotation): { pose: [0.0604616, -0.03, 1.40335, 0.980067, 0, 0.198669, 0], shape: box, size: [0.01, 0.01, 0.01] }
wide_stereo_gazebo_r_stereo_camera_frame(worldTranslationRotation): { pose: [0.0604616, -0.06, 1.40335, 0.980067, 0, 0.198669, 0], mass: 0.01, inertia: [0.001, 0.001, 0.001] }
narrow_stereo_gazebo_r_stereo_camera_frame(worldTranslationRotation): { pose: [0.0604616, -0.03, 1.40335, 0.980067, 0, 0.198669, 0], mass: 0.01, inertia: [0.001, 0.001, 0.001] }
_83(table2): { pose: [0, 0, 0.07], shape: ssBox, size: [0.27, 0.27, 0.04, 0.02], color: [0, 1, 0] }
r_shoulder_pan_joint(r_shoulder_pan_joint_origin): { pose: [0.877583, 0, 0, -0.479426], joint: hingeZ, limits: [-2.2854, 0.714602], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/shoulder_pan.h5>, gains: [220, 30], armR: True }
l_shoulder_pan_joint(l_shoulder_pan_joint_origin): { pose: [0.877583, 0, 0, 0.479426], joint: hingeZ, limits: [-0.714602, 2.2854], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/shoulder_pan.h5>, gains: [220, 30], armL: True }
r_shoulder_pan_link(r_shoulder_pan_joint): { mass: 25.7993, inertia: [0.866179, -0.0608651, -0.121181, 0.874217, -0.0588661, 0.273538] }
r_shoulder_lift_joint_origin(r_shoulder_pan_joint): { pose: [0.1, 0, 0] }
l_shoulder_pan_link(l_shoulder_pan_joint): { mass: 25.7993, inertia: [0.866179, -0.0608651, -0.121181, 0.874217, -0.0588661, 0.273538] }
l_shoulder_lift_joint_origin(l_shoulder_pan_joint): { pose: [0.1, 0, 0] }
r_shoulder_lift_joint(r_shoulder_lift_joint_origin): { pose: [0.968912, 0, 0.247404, 0], joint: hingeY, limits: [-0.5236, 1.3963], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/shoulder_lift.h5>, gains: [200, 20], armR: True }
l_shoulder_lift_joint(l_shoulder_lift_joint_origin): { pose: [0.968912, 0, 0.247404, 0], joint: hingeY, limits: [-0.5236, 1.3963], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/shoulder_lift.h5>, gains: [200, 20], armL: True }
r_shoulder_lift_link(r_shoulder_lift_joint): { mass: 2.74988, inertia: [0.0210558, 0.00496704, -0.00194809, 0.0212722, 0.00110425, 0.0197575] }
r_upper_arm_roll_joint(r_shoulder_lift_joint): { pose: [0.877583, -0.479426, 0, 0], joint: hingeX, limits: [-3.9, 0.8], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/upper_arm_roll.h5>, gains: [100, 8], armR: True }
l_shoulder_lift_link(l_shoulder_lift_joint): { mass: 2.74988, inertia: [0.0210558, 0.00496704, -0.00194809, 0.0212722, 0.00110425, 0.0197575] }
l_upper_arm_roll_joint(l_shoulder_lift_joint): { pose: [0.877583, 0.479426, 0, 0], joint: hingeX, limits: [-0.8, 3.9], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/upper_arm_roll.h5>, gains: [100, 8], armL: True }
r_upper_arm_roll_link(r_upper_arm_roll_joint): { mass: 0.1, inertia: [0.01, 0.01, 0.01] }
r_upper_arm_joint(r_upper_arm_roll_joint): { shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/upper_arm.h5> }
coll_arm_r(r_upper_arm_roll_joint): { pose: [0.221337, 0, 0], shape: ssBox, size: [0.55, 0.2, 0.2, 0.1], color: [1, 1, 1, 0.2], contact: -5, coll_pr2: True }
r_upper_arm_link(r_upper_arm_roll_joint): { mass: 6.01769, inertia: [0.0153775, 0.00375711, -0.000708529, 0.0747367, -0.000179365, 0.0760876] }
r_elbow_flex_joint_origin(r_upper_arm_roll_joint): { pose: [0.4, 0, 0] }
l_upper_arm_roll_link(l_upper_arm_roll_joint): { mass: 0.1, inertia: [0.01, 0.01, 0.01] }
l_upper_arm_joint(l_upper_arm_roll_joint): { shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/upper_arm.h5> }
coll_arm_l(l_upper_arm_roll_joint): { pose: [0.221337, 0, 0], shape: ssBox, size: [0.55, 0.2, 0.2, 0.1], color: [1, 1, 1, 0.2], contact: -5, coll_pr2: True }
l_upper_arm_link(l_upper_arm_roll_joint): { mass: 6.01769, inertia: [0.015306, -0.00339325, 0.000607655, 0.0747369, -0.000199537, 0.0760159] }
l_elbow_flex_joint_origin(l_upper_arm_roll_joint): { pose: [0.4, 0, 0] }
r_elbow_flex_joint(r_elbow_flex_joint_origin): { pose: [0.540302, 0, -0.841471, 0], joint: hingeY, limits: [-2.3213, 0], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/elbow_flex.h5>, gains: [70, 4], armR: True }
l_elbow_flex_joint(l_elbow_flex_joint_origin): { pose: [0.540302, 0, -0.841471, 0], joint: hingeY, limits: [-2.3213, 0], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/elbow_flex.h5>, gains: [70, 4], armL: True }
r_elbow_flex_link(r_elbow_flex_joint): { mass: 1.90327, inertia: [0.00346542, 4.06683e-05, 0.000431716, 0.00441606, -3.96891e-05, 0.00359157] }
r_forearm_roll_joint(r_elbow_flex_joint): { pose: [0.731689, -0.681639, 0, 0], joint: hingeX, limits: [-6, 6], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/forearm_roll.h5>, gains: [10, 1], armR: True }
l_elbow_flex_link(l_elbow_flex_joint): { mass: 1.90327, inertia: [0.00346542, 4.06683e-05, 0.000431716, 0.00441606, -3.96891e-05, 0.00359157] }
l_forearm_roll_joint(l_elbow_flex_joint): { pose: [0.731689, 0.681639, 0, 0], joint: hingeX, limits: [-6, 6], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/forearm_roll.h5>, gains: [10, 1], armL: True }
r_forearm_roll_link(r_forearm_roll_joint): { mass: 0.1, inertia: [0.01, 0.01, 0.01] }
r_forearm_joint(r_forearm_roll_joint): { shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/forearm.h5> }
coll_wrist_r(r_forearm_roll_joint): { pose: [0.21, 0, 0, 0.999391, 0, 0.0348995, 0], shape: ssBox, size: [0.35, 0.14, 0.14, 0.07], color: [1, 1, 1, 0.2], contact: -3, coll_pr2: True }
r_forearm_cam_frame_joint(r_forearm_roll_joint): { pose: [0.135, 0, 0.044, 0.679288, 0.679288, -0.196387, -0.196387], shape: box, size: [0.01, 0.01, 0.01] }
r_forearm_link(r_forearm_roll_joint): { mass: 2.57968, inertia: [0.00364857, 5.21588e-05, 0.000715348, 0.0150774, -1.31077e-05, 0.0165931] }
r_wrist_flex_joint_origin(r_forearm_roll_joint): { pose: [0.321, 0, 0] }
r_forearm_cam_frame(r_forearm_roll_joint): { pose: [0.135, 0, 0.044, 0.679288, 0.679288, -0.196387, -0.196387], mass: 0.01, inertia: [0.001, 0.001, 0.001] }
l_forearm_roll_link(l_forearm_roll_joint): { mass: 0.1, inertia: [0.01, 0.01, 0.01] }
l_forearm_joint(l_forearm_roll_joint): { shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/forearm.h5> }
coll_wrist_l(l_forearm_roll_joint): { pose: [0.21, 0, 0, 0.999391, 0, 0.0348995, 0], shape: ssBox, size: [0.35, 0.14, 0.14, 0.07], color: [1, 1, 1, 0.2], contact: -3, coll_pr2: True }
l_forearm_cam_frame_joint(l_forearm_roll_joint): { pose: [0.135, 0, 0.044, 0.679288, -0.679288, -0.196387, 0.196387], shape: box, size: [0.01, 0.01, 0.01] }
l_forearm_link(l_forearm_roll_joint): { mass: 2.57968, inertia: [0.00364857, 5.21588e-05, 0.000715348, 0.0150774, -1.31077e-05, 0.0165931] }
l_wrist_flex_joint_origin(l_forearm_roll_joint): { pose: [0.321, 0, 0] }
l_forearm_cam_frame(l_forearm_roll_joint): { pose: [0.135, 0, 0.044, 0.679288, -0.679288, -0.196387, 0.196387], mass: 0.01, inertia: [0.001, 0.001, 0.001] }
r_wrist_flex_joint(r_wrist_flex_joint_origin): { pose: [0.968912, 0, -0.247404, 0], joint: hingeY, limits: [-2.094, 0], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/wrist_flex.h5>, gains: [30, 1], armR: True }
l_wrist_flex_joint(l_wrist_flex_joint_origin): { pose: [0.968912, 0, -0.247404, 0], joint: hingeY, limits: [-2.094, 0], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/wrist_flex.h5>, gains: [30, 1], armL: True }
r_wrist_flex_link(r_wrist_flex_joint): { mass: 0.61402, inertia: [0.000651657, 2.8864e-07, 3.03477e-06, 0.000198244, -2.2645e-07, 0.000644505] }
r_wrist_roll_joint(r_wrist_flex_joint): { pose: [0.968912, -0.247404, 0, 0], joint: hingeX, limits: [-6, 6], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/wrist_roll.h5>, gains: [15, 1], armR: True }
l_wrist_flex_link(l_wrist_flex_joint): { mass: 0.61402, inertia: [0.000651657, 2.8864e-07, 3.03477e-06, 0.000198244, -2.2645e-07, 0.000644505] }
l_wrist_roll_joint(l_wrist_flex_joint): { pose: [0.968912, 0.247404, 0, 0], joint: hingeX, limits: [-6, 6], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/wrist_roll.h5>, gains: [15, 1], armL: True }
r_wrist_roll_link(r_wrist_roll_joint): { mass: 0.1, inertia: [0.01, 0.01, 0.01] }
r_gripper_palm_joint(r_wrist_roll_joint): { shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/gripper_palm.h5> }
coll_hand_r(r_wrist_roll_joint): { pose: [0.12, 0, 0], shape: ssBox, size: [0.16, 0.12, 0.06, 0.025], color: [1, 1, 1, 0.2], contact: -3, coll_pr2: True }
r_ft_sensor(r_wrist_roll_joint): { pose: [0.01, 0, 0, 0.579175, -0.405656, -0.579175, 0.405656], shape: cylinder, size: [0.0356, 0.02], color: [1, 0, 0] }
pr2R(r_wrist_roll_joint): { pose: [0.18, 0, -3.9968e-17, 0.5, 0.5, -0.5, -0.5], shape: ssBox, size: [0.03, 0.03, 0.05, 0.01], color: [1, 1, 0], logical: { is_gripper } }
r_gripper_palm_link(r_wrist_roll_joint): { mass: 0.58007, inertia: [0.000352239, -1.58048e-05, -9.175e-07, 0.000677413, -5.9554e-07, 0.000865633] }
r_gripper_motor_accelerometer_joint(r_wrist_roll_joint): { shape: box, size: [0.001, 0.001, 0.001] }
r_gripper_l_finger_joint(r_wrist_roll_joint): { pose: [0.07691, 0.01, 0, 0.99875, 0, 0, 0.0499792], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/l_finger.h5> }
r_gripper_motor_accelerometer_link(r_wrist_roll_joint): { mass: 0.001, inertia: [0.001, 0.001, 0.001] }
r_gripper_l_finger_link(r_wrist_roll_joint): { pose: [0.07691, 0.01, 0, 0.99875, 0, 0, 0.0499792], mass: 0.17126, inertia: [7.7562e-05, 1.49095e-06, -9.83385e-06, 0.000197083, -3.06125e-06, 0.000181054] }
r_gripper_r_finger_link(r_wrist_roll_joint): { pose: [0.07691, -0.01, 0, 0.99875, -0, -0, -0.0499792], mass: 0.17389, inertia: [7.73841e-05, -2.09309e-06, -8.36228e-06, 0.000198474, 2.4611e-06, 0.00018107] }
r_gripper_r_finger_link_0(r_wrist_roll_joint): { pose: [0.07691, -0.01, 0, -1.03283e-13, 0.99875, -0.0499792, 5.16845e-15], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/l_finger.h5> }
r_gripper_l_parallel_link(r_wrist_roll_joint): { pose: [0.05891, 0.031, 0, 0.99875, 0, 0, 0.0499792], mass: 0.17126, inertia: [7.7562e-05, 1.49095e-06, -9.83385e-06, 0.000197083, -3.06125e-06, 0.000181054] }
r_gripper_r_parallel_link(r_wrist_roll_joint): { pose: [0.05891, -0.031, 0, 0.99875, -0, -0, -0.0499792], mass: 0.17389, inertia: [7.73841e-05, -2.09309e-06, -8.36228e-06, 0.000198474, 2.4611e-06, 0.00018107] }
r_gripper_l_finger_tip_joint(r_wrist_roll_joint): { pose: [0.167329, 0.024047, 0, 1, 0, 0, 0], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/l_finger_tip.h5>, gripR: True }
r_gripper_l_finger_tip_link(r_wrist_roll_joint): { pose: [0.167329, 0.024047, 0], mass: 0.04419, inertia: [8.37047e-06, 5.83632e-06, 0, 9.87067e-06, 0, 1.54177e-05] }
r_gripper_r_finger_tip_link(r_wrist_roll_joint): { pose: [0.167329, -0.024047, 0], mass: 0.04419, inertia: [8.37047e-06, -5.83632e-06, 0, 9.87067e-06, 0, 1.54177e-05] }
r_gripper_r_finger_tip_link_0(r_wrist_roll_joint): { pose: [0.167329, -0.024047, 0, -1.03412e-13, 1, 0, 0], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/l_finger_tip.h5> }
l_wrist_roll_link(l_wrist_roll_joint): { mass: 0.1, inertia: [0.01, 0.01, 0.01] }
l_gripper_palm_joint(l_wrist_roll_joint): { shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/gripper_palm.h5> }
coll_hand_l(l_wrist_roll_joint): { pose: [0.12, 0, 0], shape: ssBox, size: [0.16, 0.12, 0.06, 0.025], color: [1, 1, 1, 0.2], contact: -3, coll_pr2: True }
l_ft_sensor(l_wrist_roll_joint): { pose: [0.01, 0, 0, 0.579175, -0.405656, -0.579175, 0.405656], shape: cylinder, size: [0.0356, 0.02], color: [1, 0, 0] }
pr2L(l_wrist_roll_joint): { pose: [0.18, 0, -3.9968e-17, 0.5, 0.5, -0.5, -0.5], shape: ssBox, size: [0.03, 0.03, 0.05, 0.01], color: [1, 1, 0], logical: { is_gripper } }
l_gripper_palm_link(l_wrist_roll_joint): { mass: 0.58007, inertia: [0.000352239, -1.58048e-05, -9.175e-07, 0.000677413, -5.9554e-07, 0.000865633] }
l_gripper_motor_accelerometer_joint(l_wrist_roll_joint): { shape: box, size: [0.001, 0.001, 0.001] }
l_gripper_l_finger_joint(l_wrist_roll_joint): { pose: [0.07691, 0.01, 0, 0.99875, 0, 0, 0.0499792], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/l_finger.h5> }
l_gripper_motor_accelerometer_link(l_wrist_roll_joint): { mass: 0.001, inertia: [0.001, 0.001, 0.001] }
l_gripper_l_finger_link(l_wrist_roll_joint): { pose: [0.07691, 0.01, 0, 0.99875, 0, 0, 0.0499792], mass: 0.17126, inertia: [7.7562e-05, 1.49095e-06, -9.83385e-06, 0.000197083, -3.06125e-06, 0.000181054] }
l_gripper_r_finger_link(l_wrist_roll_joint): { pose: [0.07691, -0.01, 0, 0.99875, -0, -0, -0.0499792], mass: 0.17389, inertia: [7.73841e-05, -2.09309e-06, -8.36228e-06, 0.000198474, 2.4611e-06, 0.00018107] }
l_gripper_r_finger_link_0(l_wrist_roll_joint): { pose: [0.07691, -0.01, 0, -1.03283e-13, 0.99875, -0.0499792, 5.16845e-15], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/l_finger.h5> }
l_gripper_l_parallel_link(l_wrist_roll_joint): { pose: [0.05891, 0.031, 0, 0.99875, 0, 0, 0.0499792], mass: 0.17126, inertia: [7.7562e-05, 1.49095e-06, -9.83385e-06, 0.000197083, -3.06125e-06, 0.000181054] }
l_gripper_r_parallel_link(l_wrist_roll_joint): { pose: [0.05891, -0.031, 0, 0.99875, -0, -0, -0.0499792], mass: 0.17389, inertia: [7.73841e-05, -2.09309e-06, -8.36228e-06, 0.000198474, 2.4611e-06, 0.00018107] }
l_gripper_l_finger_tip_joint(l_wrist_roll_joint): { pose: [0.167329, 0.024047, 0, 1, 0, 0, 0], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/l_finger_tip.h5>, gripL: True }
l_gripper_l_finger_tip_link(l_wrist_roll_joint): { pose: [0.167329, 0.024047, 0], mass: 0.04419, inertia: [8.37047e-06, 5.83632e-06, 0, 9.87067e-06, 0, 1.54177e-05] }
l_gripper_r_finger_tip_link(l_wrist_roll_joint): { pose: [0.167329, -0.024047, 0], mass: 0.04419, inertia: [8.37047e-06, -5.83632e-06, 0, 9.87067e-06, 0, 1.54177e-05] }
l_gripper_r_finger_tip_link_0(l_wrist_roll_joint): { pose: [0.167329, -0.024047, 0, -1.03412e-13, 1, 0, 0], shape: mesh, color: [1, 1, 1, 1], mesh: </home/mtoussai/git/rai-robotModels/pr2/meshes/l_finger_tip.h5> }

camera_init: { X: [3.2 4 2.1 -.2 .26 .76 -.56], width: 600, height: 400, focalLength: 1.5 }
