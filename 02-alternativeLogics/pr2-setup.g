world 	{  }
table1 	{  shape:ssBox size:[ 0.8 2.2 0.1 0.02 ] color:[0.3, 0.3, 0.3] contact:1 X:<0 2 0.6 0.707107 0 0 0.707107> fixed:True logical:{} }
table2 	{  shape:ssBox size:[ 0.8 2.2 0.1 0.02 ] color:[0.3, 0.3, 0.3] contact:1 X:<2 0 0.6 1 0 0 0> fixed:True logical:{table:True} }
table3 	{  shape:ssBox size:[ 0.8 2.2 0.1 0.02 ] color:[0.3, 0.3, 0.3] contact:1 X:<0 -2 0.6 0.707107 0 0 0.707107> fixed:True logical:{} }
worldTranslationRotation (world) 	{  joint:transXYPhi ctrl_H:1 limits:[ 0 0 0 1 1 1 ]  gains:[1, 1] ctrl_limits:[1, 1, 1] base:True }
obj0 (table1) 	{  joint:rigid ctrl_H:1  shape:ssBox size:[ 0.1 0.1 0.2 0.02 ] color:[1, 0, 0] contact:1 Q:<-0.0249071 -0.0437994 0.15 0.48503 0 0 -0.874497> logical:{object:True} }
obj1 (table1) 	{  joint:rigid ctrl_H:1  shape:ssBox size:[ 0.1 0.1 0.2 0.02 ] color:[1, 0, 0] contact:1 Q:<-0.0991834 0.83586 0.15 0.514909 0 0 0.857245> logical:{object:True} }
obj2 (table1) 	{  joint:rigid ctrl_H:1  shape:ssBox size:[ 0.1 0.1 0.2 0.02 ] color:[1, 0, 0] contact:1 Q:<0.211826 0.124012 0.15 0.971684 0 0 0.236284> logical:{object:True} }
obj3 (table1) 	{  joint:rigid ctrl_H:1  shape:ssBox size:[ 0.1 0.1 0.2 0.02 ] color:[1, 0, 0] contact:1 Q:<0.0385723 -0.890782 0.15 0.465715 0 0 0.884935> logical:{object:True} }
tray (table2) 	{  shape:ssBox size:[ 0.15 0.15 0.04 0.02 ] color:[0, 1, 0] Q:<0 0 0.07 1 0 0 0> logical:{table:True} }
_10 (table2) 	{  shape:ssBox size:[ 0.27 0.27 0.04 0.02 ] color:[0, 1, 0] Q:<0 0 0.07 1 0 0 0> }
base_footprint (worldTranslationRotation) 	{  mass:140.967 }
base_link_1 (worldTranslationRotation) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/base_v0/base.stl' meshscale:0.1 Q:<-0.0282159 0.00517696 0.272424 1 0 0 0> rel_includes_mesh_center:True }
base_footprint_1 (worldTranslationRotation) 	{  shape:box size:[ 0.01 0.01 0.01 0 ] }
base_bellow_link_1 (worldTranslationRotation) 	{  shape:box size:[ 0.05 0.37 0.3 0 ] Q:<-0.29 0 0.851 1 0 0 0> }
fl_caster_rotation_link_1 (worldTranslationRotation) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/base_v0/caster.stl' Q:<0.21838 0.225407 0.180919 1 0 0 0> rel_includes_mesh_center:True }
fl_caster_l_wheel_link_1 (worldTranslationRotation) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/base_v0/wheel.stl' Q:<0.224225 0.270791 0.0789213 1 0 0 0> rel_includes_mesh_center:True }
fl_caster_r_wheel_link_1 (worldTranslationRotation) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/base_v0/wheel.stl' Q:<0.224225 0.172791 0.0789213 1 0 0 0> rel_includes_mesh_center:True }
fr_caster_rotation_link_1 (worldTranslationRotation) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/base_v0/caster.stl' Q:<0.21838 -0.223793 0.180919 1 0 0 0> rel_includes_mesh_center:True }
fr_caster_l_wheel_link_1 (worldTranslationRotation) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/base_v0/wheel.stl' Q:<0.224225 -0.178409 0.0789213 1 0 0 0> rel_includes_mesh_center:True }
fr_caster_r_wheel_link_1 (worldTranslationRotation) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/base_v0/wheel.stl' Q:<0.224225 -0.276409 0.0789213 1 0 0 0> rel_includes_mesh_center:True }
bl_caster_rotation_link_1 (worldTranslationRotation) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/base_v0/caster.stl' Q:<-0.23082 0.225407 0.180919 1 0 0 0> rel_includes_mesh_center:True }
bl_caster_l_wheel_link_1 (worldTranslationRotation) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/base_v0/wheel.stl' Q:<-0.224975 0.270791 0.0789213 1 0 0 0> rel_includes_mesh_center:True }
bl_caster_r_wheel_link_1 (worldTranslationRotation) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/base_v0/wheel.stl' Q:<-0.224975 0.172791 0.0789213 1 0 0 0> rel_includes_mesh_center:True }
br_caster_rotation_link_1 (worldTranslationRotation) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/base_v0/caster.stl' Q:<-0.23082 -0.223793 0.180919 1 0 0 0> rel_includes_mesh_center:True }
br_caster_l_wheel_link_1 (worldTranslationRotation) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/base_v0/wheel.stl' Q:<-0.224975 -0.178409 0.0789213 1 0 0 0> rel_includes_mesh_center:True }
br_caster_r_wheel_link_1 (worldTranslationRotation) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/base_v0/wheel.stl' Q:<-0.224975 -0.276409 0.0789213 1 0 0 0> rel_includes_mesh_center:True }
coll_base (worldTranslationRotation) 	{  shape:ssBox size:[ 0.7 0.7 0.36 0.1 ] color:[0.5, 0, 0, 0.2] contact:-2 Q:<0 0 0.18 1 0 0 0> coll_pr2:True }
coll_torso (worldTranslationRotation) 	{  shape:ssBox size:[ 0.45 0.7 1.1 0.1 ] color:[0.5, 0, 0, 0.2] contact:-2 Q:<-0.13 0 0.55 1 0 0 0> coll_pr2:True }
base_footprint (worldTranslationRotation) 	{  shape:marker size:[ 0.5 0 0 0 ] color:[1, 0, 0] }
base_footprint>torso_lift_joint (worldTranslationRotation) 	{  Q:<-0.05 0 0.790675 0.707107 0 -0.707107 0> }
torso_lift_joint (base_footprint>torso_lift_joint) 	{  joint:transX ctrl_H:3000 limits:[ 0.0115 0.325 0.013 10000 1 ]  Q:<0.1 0 0 1 0 0 0> ctrl_limits:[0.013, 10000, 1] gains:[100000, 10] torso:True }
torso_lift_link (torso_lift_joint) 	{  mass:36.449 }
torso_lift_link_1 (torso_lift_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/torso_v0/torso_lift.stl' Q:<0.120676 -0.00108579 0.0665629 -0.707107 0 -0.707107 0> rel_includes_mesh_center:True }
endeffBase (torso_lift_joint) 	{  shape:marker size:[ 0.1 0 0 0 ] color:[1, 0, 0] Q:<4.44089e-17 0 -0.2 0.707107 0 0.707107 0> }
endeffWorkspace (torso_lift_joint) 	{  shape:marker size:[ 0.1 0 0 0 ] color:[1, 0, 0] Q:<-0.1 0 -0.6 0.5 -0.5 0.5 -0.5> }
torso_lift_link>head_pan_joint (torso_lift_joint) 	{  Q:<0.38145 0 0.01707 1 0 0 0> }
torso_lift_link>laser_tilt_mount_joint (torso_lift_joint) 	{  Q:<0.227 0 -0.09893 -0.5 -0.5 -0.5 -0.5> }
torso_lift_link>r_shoulder_pan_joint (torso_lift_joint) 	{  Q:<0 -0.188 0 1 0 0 0> }
torso_lift_link>l_shoulder_pan_joint (torso_lift_joint) 	{  Q:<0 0.188 0 1 0 0 0> }
head_pan_joint (torso_lift_link>head_pan_joint) 	{  joint:hingeX ctrl_H:0.1 limits:[ -2.857 2.857 6 2.645 0.5 ]  ctrl_limits:[6, 2.645, 0.5] gains:[20, 2] head:True }
laser_tilt_mount_joint (torso_lift_link>laser_tilt_mount_joint) 	{  joint:hingeX ctrl_H:1 limits:[ -0.7354 1.43353 10 0.65 1 ]  ctrl_limits:[10, 0.65, 1] laser:True }
r_shoulder_pan_joint (torso_lift_link>r_shoulder_pan_joint) 	{  joint:hingeX ctrl_H:1 limits:[ -2.1354 0.564602 2.088 30 4 ]  Q:<0 0 0 0.877583 -0.479426 0 0> ctrl_limits:[2.088, 30, 4] gains:[220, 30] armR:True }
l_shoulder_pan_joint (torso_lift_link>l_shoulder_pan_joint) 	{  joint:hingeX ctrl_H:1 limits:[ -0.564602 2.1354 2.088 30 4 ]  Q:<0 0 0 0.877583 0.479426 0 0> ctrl_limits:[2.088, 30, 4] gains:[220, 30] armL:True }
head_pan_link (head_pan_joint) 	{  mass:6.339 }
head_pan_link_1 (head_pan_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/head_v0/head_pan.stl' Q:<-0.00764864 0.000777182 -0.0404273 -0.707107 0 -0.707107 0> rel_includes_mesh_center:True }
head_pan_link>head_tilt_joint (head_pan_joint) 	{  Q:<1.5099e-17 0 -0.068 -0.5 -0.5 -0.5 -0.5> }
laser_tilt_mount_link (laser_tilt_mount_joint) 	{  mass:0.592 }
laser_tilt_mount_link_1 (laser_tilt_mount_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/tilting_laser_v0/tilting_hokuyo.stl' Q:<-0.00300578 0.00167121 0.0122175 -0.707107 0 0 0.707107> rel_includes_mesh_center:True }
endeffLaser (laser_tilt_mount_joint) 	{  shape:marker size:[ 0.1 0 0 0 ] color:[1, 1, 1] Q:<-0.00300578 0.00167121 0.0122175 4.32978e-17 -4.32978e-17 -0.707107 0.707107> }
r_shoulder_pan_link (r_shoulder_pan_joint) 	{  mass:25.7993 }
r_shoulder_pan_link_1 (r_shoulder_pan_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/shoulder_v0/shoulder_pan.stl' Q:<-0.16813 0.00258043 -0.00550141 -0.707107 0 -0.707107 0> rel_includes_mesh_center:True }
r_shoulder_pan_link>r_shoulder_lift_joint (r_shoulder_pan_joint) 	{  Q:<2.22045e-17 0 -0.1 -0.5 -0.5 -0.5 -0.5> }
l_shoulder_pan_link (l_shoulder_pan_joint) 	{  mass:25.7993 }
l_shoulder_pan_link_1 (l_shoulder_pan_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/shoulder_v0/shoulder_pan.stl' Q:<-0.16813 0.00258043 -0.00550141 -0.707107 0 -0.707107 0> rel_includes_mesh_center:True }
l_shoulder_pan_link>l_shoulder_lift_joint (l_shoulder_pan_joint) 	{  Q:<2.22045e-17 0 -0.1 -0.5 -0.5 -0.5 -0.5> }
head_tilt_joint (head_pan_link>head_tilt_joint) 	{  joint:hingeX ctrl_H:0.1 limits:[ -0.3712 1.29626 5 18 0.1 ]  Q:<0 0 0 0.980067 0.198669 0 0> ctrl_limits:[5, 18, 0.1] gains:[60, 4] head:True }
r_shoulder_lift_joint (r_shoulder_pan_link>r_shoulder_lift_joint) 	{  joint:hingeX ctrl_H:1 limits:[ -0.3536 1.2963 2.082 30 4 ]  Q:<0 0 0 0.968912 0.247404 0 0> ctrl_limits:[2.082, 30, 4] gains:[200, 20] armR:True }
l_shoulder_lift_joint (l_shoulder_pan_link>l_shoulder_lift_joint) 	{  joint:hingeX ctrl_H:1 limits:[ -0.3536 1.2963 2.082 30 4 ]  Q:<0 0 0 0.968912 0.247404 0 0> ctrl_limits:[2.082, 30, 4] gains:[200, 20] armL:True }
head_tilt_link (head_tilt_joint) 	{  mass:5.441 }
head_tilt_link_1 (head_tilt_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/head_v0/head_tilt.stl' Q:<0.000961809 -0.00114119 0.0964492 -0.707107 0 0 0.707107> rel_includes_mesh_center:True }
head_plate_frame_1 (head_tilt_joint) 	{  shape:box size:[ 0.01 0.01 0.01 0 ] Q:<5.15143e-18 -0.0232 0.0645 -0.707107 0 0 0.707107> }
sensor_mount_link_1 (head_tilt_joint) 	{  shape:box size:[ 0.01 0.01 0.01 0 ] Q:<5.15143e-18 -0.0232 0.0645 -0.707107 0 0 0.707107> }
double_stereo_link_1 (head_tilt_joint) 	{  shape:box size:[ 0.02 0.12 0.05 0 ] Q:<2.93099e-18 -0.0132 0.0895 -0.707107 0 0 0.707107> }
head_mount_link_1 (head_tilt_joint) 	{  shape:mesh color:[0.5, 0.5, 0.5, 1] mesh:'../../rai-robotModels/pr2/sensors/kinect_prosilica_v0/115x100_swept_back--coarse.stl' meshscale:0.001 Q:<0.00394516 0.168922 0.258886 -0.707107 0 0 0.707107> rel_includes_mesh_center:True }
head_mount_kinect_ir_link_1 (head_tilt_joint) 	{  shape:sphere size:[ 0 0 0 0.0005 ] Q:<0.0125 0.147067 0.291953 -0.707107 0 0 0.707107> }
head_mount_kinect_rgb_link_1 (head_tilt_joint) 	{  shape:sphere size:[ 0 0 0 0.0005 ] Q:<-0.0175 0.147067 0.291953 -0.707107 0 0 0.707107> }
head_mount_prosilica_link_1 (head_tilt_joint) 	{  shape:sphere size:[ 0 0 0 0.0005 ] Q:<0.0125 0.161257 0.244421 -0.707107 0 0 0.707107> }
endeffHead (head_tilt_joint) 	{  shape:marker size:[ 0.1 0 0 0 ] color:[1, 0, 0] Q:<1.77636e-17 -0.08 0.12 1.11022e-16 0 -0.707107 -0.707107> }
endeffEyes (head_tilt_joint) 	{  shape:marker size:[ 0.1 0 0 0 ] color:[1, 1, 0] Q:<0 -0.05 0.12 4.32978e-17 -4.32978e-17 -0.707107 0.707107> }
endeffKinect (head_tilt_joint) 	{  shape:marker size:[ 0.1 0 0 0 ] color:[1, 0, 0] Q:<-0.0175 0.147067 0.291953 4.32978e-17 4.32978e-17 0.707107 0.707107> }
r_shoulder_lift_link (r_shoulder_lift_joint) 	{  mass:2.74988 }
r_shoulder_lift_link_1 (r_shoulder_lift_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/shoulder_v0/shoulder_lift.stl' Q:<-0.00127619 -0.0563251 0.0161388 -0.707107 0 0 0.707107> rel_includes_mesh_center:True }
r_shoulder_lift_link>r_upper_arm_roll_joint (r_shoulder_lift_joint) 	{  Q:<0 0 0 -0.707107 0 0 0.707107> }
l_shoulder_lift_link (l_shoulder_lift_joint) 	{  mass:2.74988 }
l_shoulder_lift_link_1 (l_shoulder_lift_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/shoulder_v0/shoulder_lift.stl' Q:<-0.00127619 -0.0563251 0.0161388 -0.707107 0 0 0.707107> rel_includes_mesh_center:True }
l_shoulder_lift_link>l_upper_arm_roll_joint (l_shoulder_lift_joint) 	{  Q:<0 0 0 -0.707107 0 0 0.707107> }
r_upper_arm_roll_joint (r_shoulder_lift_link>r_upper_arm_roll_joint) 	{  joint:hingeX ctrl_H:1 limits:[ -3.75 0.65 3.27 30 4 ]  Q:<0 0 0 0.877583 -0.479426 0 0> ctrl_limits:[3.27, 30, 4] gains:[100, 8] armR:True }
l_upper_arm_roll_joint (l_shoulder_lift_link>l_upper_arm_roll_joint) 	{  joint:hingeX ctrl_H:1 limits:[ -0.65 3.75 3.27 30 4 ]  Q:<0 0 0 0.877583 0.479426 0 0> ctrl_limits:[3.27, 30, 4] gains:[100, 8] armL:True }
r_upper_arm_roll_link (r_upper_arm_roll_joint) 	{  mass:6.11769 }
r_upper_arm_roll_link_1 (r_upper_arm_roll_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/shoulder_v0/upper_arm_roll.stl' Q:<0.121137 9.59109e-05 5.64062e-05 1 0 0 0> rel_includes_mesh_center:True }
r_upper_arm_link_1 (r_upper_arm_roll_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/upper_arm_v0/upper_arm.stl' Q:<0.303332 -0.00060982 -0.0039943 1 0 0 0> rel_includes_mesh_center:True }
coll_arm_r (r_upper_arm_roll_joint) 	{  shape:ssBox size:[ 0.55 0.2 0.2 0.1 ] color:[0.5, 0, 0, 0.2] contact:-4 Q:<0.221337 0 0 1 0 0 0> coll_pr2:True }
r_upper_arm_roll_link>r_elbow_flex_joint (r_upper_arm_roll_joint) 	{  Q:<0.4 0 0 0.707107 0 0 0.707107> }
l_upper_arm_roll_link (l_upper_arm_roll_joint) 	{  mass:6.11769 }
l_upper_arm_roll_link_1 (l_upper_arm_roll_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/shoulder_v0/upper_arm_roll.stl' Q:<0.121137 9.59109e-05 5.64062e-05 1 0 0 0> rel_includes_mesh_center:True }
l_upper_arm_link_1 (l_upper_arm_roll_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/upper_arm_v0/upper_arm.stl' Q:<0.303332 -0.00060982 -0.0039943 1 0 0 0> rel_includes_mesh_center:True }
coll_arm_l (l_upper_arm_roll_joint) 	{  shape:ssBox size:[ 0.55 0.2 0.2 0.1 ] color:[0.5, 0, 0, 0.2] contact:-4 Q:<0.221337 0 0 1 0 0 0> coll_pr2:True }
l_upper_arm_roll_link>l_elbow_flex_joint (l_upper_arm_roll_joint) 	{  Q:<0.4 0 0 0.707107 0 0 0.707107> }
r_elbow_flex_joint (r_upper_arm_roll_link>r_elbow_flex_joint) 	{  joint:hingeX ctrl_H:1 limits:[ -2.1213 -0.15 3.3 30 4 ]  Q:<0 0 0 0.540302 -0.841471 0 0> ctrl_limits:[3.3, 30, 4] gains:[70, 4] armR:True }
l_elbow_flex_joint (l_upper_arm_roll_link>l_elbow_flex_joint) 	{  joint:hingeX ctrl_H:1 limits:[ -2.1213 -0.15 3.3 30 4 ]  Q:<0 0 0 0.540302 -0.841471 0 0> ctrl_limits:[3.3, 30, 4] gains:[70, 4] armL:True }
r_elbow_flex_link (r_elbow_flex_joint) 	{  mass:1.90327 }
r_elbow_flex_link_1 (r_elbow_flex_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/upper_arm_v0/elbow_flex.stl' Q:<-0.00060554 -0.0250394 -0.00341596 -0.707107 0 0 0.707107> rel_includes_mesh_center:True }
r_elbow_flex_link>r_forearm_roll_joint (r_elbow_flex_joint) 	{  Q:<0 0 0 -0.707107 0 0 0.707107> }
l_elbow_flex_link (l_elbow_flex_joint) 	{  mass:1.90327 }
l_elbow_flex_link_1 (l_elbow_flex_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/upper_arm_v0/elbow_flex.stl' Q:<-0.00060554 -0.0250394 -0.00341596 -0.707107 0 0 0.707107> rel_includes_mesh_center:True }
l_elbow_flex_link>l_forearm_roll_joint (l_elbow_flex_joint) 	{  Q:<0 0 0 -0.707107 0 0 0.707107> }
r_forearm_roll_joint (r_elbow_flex_link>r_forearm_roll_joint) 	{  joint:hingeX ctrl_H:1 limits:[ 0 0 0 3.6 30 2 ]  Q:<0 0 0 0.731689 -0.681639 0 0> ctrl_limits:[3.6, 30, 2] gains:[10, 1] armR:True }
l_forearm_roll_joint (l_elbow_flex_link>l_forearm_roll_joint) 	{  joint:hingeX ctrl_H:1 limits:[ 0 0 0 3.6 30 2 ]  Q:<0 0 0 0.731689 0.681639 0 0> ctrl_limits:[3.6, 30, 2] gains:[10, 1] armL:True }
r_forearm_roll_link (r_forearm_roll_joint) 	{  mass:2.68968 }
r_forearm_roll_link_1 (r_forearm_roll_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/upper_arm_v0/forearm_roll.stl' Q:<0.086794 -0.000500601 0.00973495 1 0 0 0> rel_includes_mesh_center:True }
r_forearm_link_1 (r_forearm_roll_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/forearm_v0/forearm.stl' Q:<0.216445 0.000691519 0.00300974 1 0 0 0> rel_includes_mesh_center:True }
coll_wrist_r (r_forearm_roll_joint) 	{  shape:ssBox size:[ 0.35 0.14 0.14 0.07 ] color:[0.5, 0, 0, 0.2] contact:-2 Q:<0.21 0 0 0.999391 0 0.0348995 0> coll_pr2:True }
r_forearm_roll_link>r_wrist_flex_joint (r_forearm_roll_joint) 	{  Q:<0.321 0 0 0.707107 0 0 0.707107> }
l_forearm_roll_link (l_forearm_roll_joint) 	{  mass:2.68968 }
l_forearm_roll_link_1 (l_forearm_roll_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/upper_arm_v0/forearm_roll.stl' Q:<0.086794 -0.000500601 0.00973495 1 0 0 0> rel_includes_mesh_center:True }
l_forearm_link_1 (l_forearm_roll_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/forearm_v0/forearm.stl' Q:<0.216445 0.000691519 0.00300974 1 0 0 0> rel_includes_mesh_center:True }
coll_wrist_l (l_forearm_roll_joint) 	{  shape:ssBox size:[ 0.35 0.14 0.14 0.07 ] color:[0.5, 0, 0, 0.2] contact:-2 Q:<0.21 0 0 0.999391 0 0.0348995 0> coll_pr2:True }
l_forearm_roll_link>l_wrist_flex_joint (l_forearm_roll_joint) 	{  Q:<0.321 0 0 0.707107 0 0 0.707107> }
r_wrist_flex_joint (r_forearm_roll_link>r_wrist_flex_joint) 	{  joint:hingeX ctrl_H:1 limits:[ -2 -0.1 3.078 10 2 ]  Q:<0 0 0 0.968912 -0.247404 0 0> ctrl_limits:[3.078, 10, 2] gains:[30, 1] armR:True }
l_wrist_flex_joint (l_forearm_roll_link>l_wrist_flex_joint) 	{  joint:hingeX ctrl_H:1 limits:[ -2 -0.1 3.078 10 2 ]  Q:<0 0 0 0.968912 -0.247404 0 0> ctrl_limits:[3.078, 10, 2] gains:[30, 1] armL:True }
r_wrist_flex_link (r_wrist_flex_joint) 	{  mass:0.61402 }
r_wrist_flex_link_1 (r_wrist_flex_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/forearm_v0/wrist_flex.stl' Q:<-0.000233081 0.00258595 -0.00218093 -0.707107 0 0 0.707107> rel_includes_mesh_center:True }
r_wrist_flex_link>r_wrist_roll_joint (r_wrist_flex_joint) 	{  Q:<0 0 0 -0.707107 0 0 0.707107> }
l_wrist_flex_link (l_wrist_flex_joint) 	{  mass:0.61402 }
l_wrist_flex_link_1 (l_wrist_flex_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/forearm_v0/wrist_flex.stl' Q:<-0.000233081 0.00258595 -0.00218093 -0.707107 0 0 0.707107> rel_includes_mesh_center:True }
l_wrist_flex_link>l_wrist_roll_joint (l_wrist_flex_joint) 	{  Q:<0 0 0 -0.707107 0 0 0.707107> }
r_wrist_roll_joint (r_wrist_flex_link>r_wrist_roll_joint) 	{  joint:hingeX ctrl_H:1 limits:[ 0 0 0 3.6 10 2 ]  Q:<0 0 0 0.968912 -0.247404 0 0> ctrl_limits:[3.6, 10, 2] gains:[15, 1] armR:True }
l_wrist_roll_joint (l_wrist_flex_link>l_wrist_roll_joint) 	{  joint:hingeX ctrl_H:1 limits:[ 0 0 0 3.6 10 2 ]  Q:<0 0 0 0.968912 0.247404 0 0> ctrl_limits:[3.6, 10, 2] gains:[15, 1] armL:True }
r_wrist_roll_link (r_wrist_roll_joint) 	{  mass:0.681071 Q:<0.0356 0 0 1 0 0 0> }
r_wrist_roll_link_1 (r_wrist_roll_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/forearm_v0/wrist_roll.stl' Q:<0.0673264 0.000290217 -0.00107323 1 0 0 0> rel_includes_mesh_center:True }
r_gripper_palm_link_1 (r_wrist_roll_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/gripper_v0/gripper_palm.stl' Q:<0.123996 0.000221324 -2.62985e-05 1 0 0 0> rel_includes_mesh_center:True }
r_gripper_motor_accelerometer_link_1 (r_wrist_roll_joint) 	{  shape:box size:[ 0.001 0.001 0.001 0 ] Q:<0.0356 0 0 1 0 0 0> }
coll_hand_r (r_wrist_roll_joint) 	{  shape:ssBox size:[ 0.16 0.12 0.06 0.025 ] color:[0.5, 0, 0, 0.2] contact:-2 Q:<0.1556 0 0 1 0 0 0> coll_pr2:True }
r_ft_sensor (r_wrist_roll_joint) 	{  shape:cylinder size:[ 0 0 0.0356 0.02 ] color:[1, 0, 0] Q:<0.0456 0 0 0.579175 -0.405656 -0.579175 0.405656> }
pr2R (r_wrist_roll_joint) 	{  shape:ssBox size:[ 0.03 0.03 0.05 0.01 ] color:[1, 1, 0] Q:<0.2156 0 -3.9968e-17 0.5 0.5 -0.5 -0.5> logical:{gripper:True} logical:{gripper:True} }
r_wrist_roll_link>r_gripper_l_finger_joint (r_wrist_roll_joint) 	{  Q:<0.11251 0.01 0 0.707107 0 -0.707107 0> }
r_wrist_roll_link>r_gripper_r_finger_joint (r_wrist_roll_joint) 	{  Q:<0.11251 -0.01 0 0.707107 -0 0.707107 0> }
l_wrist_roll_link (l_wrist_roll_joint) 	{  mass:0.681071 Q:<0.0356 0 0 1 0 0 0> }
l_wrist_roll_link_1 (l_wrist_roll_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/forearm_v0/wrist_roll.stl' Q:<0.0673264 0.000290217 -0.00107323 1 0 0 0> rel_includes_mesh_center:True }
l_gripper_palm_link_1 (l_wrist_roll_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/gripper_v0/gripper_palm.stl' Q:<0.123996 0.000221324 -2.62985e-05 1 0 0 0> rel_includes_mesh_center:True }
l_gripper_motor_accelerometer_link_1 (l_wrist_roll_joint) 	{  shape:box size:[ 0.001 0.001 0.001 0 ] Q:<0.0356 0 0 1 0 0 0> }
coll_hand_l (l_wrist_roll_joint) 	{  shape:ssBox size:[ 0.16 0.12 0.06 0.025 ] color:[0.5, 0, 0, 0.2] contact:-2 Q:<0.1556 0 0 1 0 0 0> coll_pr2:True }
l_ft_sensor (l_wrist_roll_joint) 	{  shape:cylinder size:[ 0 0 0.0356 0.02 ] color:[1, 0, 0] Q:<0.0456 0 0 0.579175 -0.405656 -0.579175 0.405656> }
pr2L (l_wrist_roll_joint) 	{  shape:ssBox size:[ 0.03 0.03 0.05 0.01 ] color:[1, 1, 0] Q:<0.2156 0 -3.9968e-17 0.5 0.5 -0.5 -0.5> logical:{gripper:True} logical:{gripper:True} }
l_wrist_roll_link>l_gripper_l_finger_joint (l_wrist_roll_joint) 	{  Q:<0.11251 0.01 0 0.707107 0 -0.707107 0> }
l_wrist_roll_link>l_gripper_r_finger_joint (l_wrist_roll_joint) 	{  Q:<0.11251 -0.01 0 0.707107 -0 0.707107 0> }
r_gripper_l_finger_joint (r_wrist_roll_link>r_gripper_l_finger_joint) 	{  joint:hingeX ctrl_H:1 limits:[ 0 0.548 0.5 1000 1 ]  Q:<0 0 0 0.99875 0.0499792 0 0> ctrl_limits:[0.5, 1000, 1] gripR:True }
r_gripper_r_finger_joint (r_wrist_roll_link>r_gripper_r_finger_joint) 	{  joint:hingeX ctrl_H:1 limits:[ 0 0.548 0.5 1000 1 ] mimic:(r_gripper_l_finger_joint)  ctrl_limits:[0.5, 1000, 1] gripR:True }
l_gripper_l_finger_joint (l_wrist_roll_link>l_gripper_l_finger_joint) 	{  joint:hingeX ctrl_H:1 limits:[ 0 0.548 0.5 1000 1 ]  Q:<0 0 0 0.99875 0.0499792 0 0> ctrl_limits:[0.5, 1000, 1] gripL:True }
l_gripper_r_finger_joint (l_wrist_roll_link>l_gripper_r_finger_joint) 	{  joint:hingeX ctrl_H:1 limits:[ 0 0.548 0.5 1000 1 ] mimic:(l_gripper_l_finger_joint)  ctrl_limits:[0.5, 1000, 1] gripL:True }
r_gripper_l_finger_link (r_gripper_l_finger_joint) 	{  mass:0.17126 }
r_gripper_l_finger_link_1 (r_gripper_l_finger_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/gripper_v0/l_finger.stl' Q:<-0.000214812 0.0125558 -0.0493868 -0.707107 0 -0.707107 0> rel_includes_mesh_center:True }
r_gripper_l_finger_link>r_gripper_l_finger_tip_joint (r_gripper_l_finger_joint) 	{  Q:<2.02882e-17 0.00495 -0.09137 -2.22045e-16 0 -1 0> }
r_gripper_r_finger_link (r_gripper_r_finger_joint) 	{  mass:0.17389 }
r_gripper_r_finger_link_1 (r_gripper_r_finger_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/gripper_v0/l_finger.stl' Q:<-0.000214812 -0.0125558 0.0493868 7.3123e-14 -0.707107 -7.3123e-14 -0.707107> rel_includes_mesh_center:True }
r_gripper_r_finger_link>r_gripper_r_finger_tip_joint (r_gripper_r_finger_joint) 	{  Q:<2.02882e-17 -0.00495 0.09137 -2.22045e-16 0 1 0> }
l_gripper_l_finger_link (l_gripper_l_finger_joint) 	{  mass:0.17126 }
l_gripper_l_finger_link_1 (l_gripper_l_finger_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/gripper_v0/l_finger.stl' Q:<-0.000214812 0.0125558 -0.0493868 -0.707107 0 -0.707107 0> rel_includes_mesh_center:True }
l_gripper_l_finger_link>l_gripper_l_finger_tip_joint (l_gripper_l_finger_joint) 	{  Q:<2.02882e-17 0.00495 -0.09137 -2.22045e-16 0 -1 0> }
l_gripper_r_finger_link (l_gripper_r_finger_joint) 	{  mass:0.17389 }
l_gripper_r_finger_link_1 (l_gripper_r_finger_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/gripper_v0/l_finger.stl' Q:<-0.000214812 -0.0125558 0.0493868 7.3123e-14 -0.707107 -7.3123e-14 -0.707107> rel_includes_mesh_center:True }
l_gripper_r_finger_link>l_gripper_r_finger_tip_joint (l_gripper_r_finger_joint) 	{  Q:<2.02882e-17 -0.00495 0.09137 -2.22045e-16 0 1 0> }
r_gripper_l_finger_tip_joint (r_gripper_l_finger_link>r_gripper_l_finger_tip_joint) 	{  joint:hingeX ctrl_H:1 limits:[ 0 0.548 0.5 1000 1 ] mimic:(r_gripper_l_finger_joint)  ctrl_limits:[0.5, 1000, 1] gripR:True }
r_gripper_r_finger_tip_joint (r_gripper_r_finger_link>r_gripper_r_finger_tip_joint) 	{  joint:hingeX ctrl_H:1 limits:[ 0 0.548 0.5 1000 1 ] mimic:(r_gripper_l_finger_joint)  ctrl_limits:[0.5, 1000, 1] gripR:True }
l_gripper_l_finger_tip_joint (l_gripper_l_finger_link>l_gripper_l_finger_tip_joint) 	{  joint:hingeX ctrl_H:1 limits:[ 0 0.548 0.5 1000 1 ] mimic:(l_gripper_l_finger_joint)  ctrl_limits:[0.5, 1000, 1] gripL:True }
l_gripper_r_finger_tip_joint (l_gripper_r_finger_link>l_gripper_r_finger_tip_joint) 	{  joint:hingeX ctrl_H:1 limits:[ 0 0.548 0.5 1000 1 ] mimic:(l_gripper_l_finger_joint)  ctrl_limits:[0.5, 1000, 1] gripL:True }
r_gripper_l_finger_tip_link (r_gripper_l_finger_tip_joint) 	{  mass:0.04419 }
r_gripper_l_finger_tip_link_1 (r_gripper_l_finger_tip_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/gripper_v0/l_finger_tip.stl' Q:<0.000126401 0.000750209 0.0081309 -0.707107 -0 0.707107 0> rel_includes_mesh_center:True }
r_gripper_r_finger_tip_link (r_gripper_r_finger_tip_joint) 	{  mass:0.04419 }
r_gripper_r_finger_tip_link_1 (r_gripper_r_finger_tip_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/gripper_v0/l_finger_tip.stl' Q:<0.000126401 -0.000750209 -0.0081309 7.3123e-14 -0.707107 7.3123e-14 0.707107> rel_includes_mesh_center:True }
r_gripper_r_finger_tip_link>r_gripper_joint (r_gripper_r_finger_tip_joint) 	{  Q:<0 0 0 -0.5 -0.5 -0.5 -0.5> }
l_gripper_l_finger_tip_link (l_gripper_l_finger_tip_joint) 	{  mass:0.04419 }
l_gripper_l_finger_tip_link_1 (l_gripper_l_finger_tip_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/gripper_v0/l_finger_tip.stl' Q:<0.000126401 0.000750209 0.0081309 -0.707107 -0 0.707107 0> rel_includes_mesh_center:True }
l_gripper_r_finger_tip_link (l_gripper_r_finger_tip_joint) 	{  mass:0.04419 }
l_gripper_r_finger_tip_link_1 (l_gripper_r_finger_tip_joint) 	{  shape:mesh mesh:'../../rai-robotModels/pr2/gripper_v0/l_finger_tip.stl' Q:<0.000126401 -0.000750209 -0.0081309 7.3123e-14 -0.707107 7.3123e-14 0.707107> rel_includes_mesh_center:True }
l_gripper_r_finger_tip_link>l_gripper_joint (l_gripper_r_finger_tip_joint) 	{  Q:<0 0 0 -0.5 -0.5 -0.5 -0.5> }
r_gripper_joint (r_gripper_r_finger_tip_link>r_gripper_joint) 	{  joint:transX ctrl_H:1 limits:[ -0.01 0.088 0.2 1000 1 ]  Q:<0.01 0 0 1 0 0 0> ctrl_limits:[0.2, 1000, 1] gains:[1000, 1] gripR:True }
l_gripper_joint (l_gripper_r_finger_tip_link>l_gripper_joint) 	{  joint:transX ctrl_H:1 limits:[ -0.01 0.088 0.2 1000 1 ]  Q:<0.01 0 0 1 0 0 0> ctrl_limits:[0.2, 1000, 1] gains:[1000, 1] gripL:True }
r_gripper_l_finger_tip_frame (r_gripper_joint) 	{  mass:0.05 }
r_gripper_frame (r_gripper_joint) 	{  shape:marker size:[ 0.1 0 0 0 ] Q:<0 0 0 -0.707107 0 0 0.707107> }
l_gripper_l_finger_tip_frame (l_gripper_joint) 	{  mass:0.05 }
l_gripper_frame (l_gripper_joint) 	{  shape:marker size:[ 0.1 0 0 0 ] Q:<0 0 0 -0.707107 0 0 0.707107> }

