World 	{  X: [0, 0, 0, 1, 0, 0, 0] }                         

table_base (World) {
    Q:[0 0 .6]
    shape:marker, size:[.03],
}

table (table_base){
    shape:ssBox, Q:[0 0 -.1], size:[2 2 .1 .02], color:[.7 .7 .7]
    contact, logical: { is_place }
}

agent (table_base){ joint:transXY limits: [-1 -1 1 1], type:ssCylinder, size:[0.1 0.15 .01], contact:1 Q: [  -0.3, -.3, 0., 1, 0, .0, .0] color:[0.1, 0.1, 0.9, 1] logical: { is_gripper } }
obj (table){ joint:rigid type:ssBox, size:[0.3 0.3 0.1 .01], contact:1 Q: [  0.75, 0.4, 0.1, 1, 0, .0, .0] color:[0.9, 0.1, 0.1, 1]}
goal (table_base){ type:ssBox, size:[0.2 0.2 0.1 .005], contact:0 Q: [  -0.75, .4, 0.0, 1, 0, .0, .0] color:[0.4, 1, 0.4, 0.3], logical: { is_pose } }

_obs_top_right (table_base){  type:ssBox, size:[0.5 1.2 0.1 .005], contact:1 Q: [  -0.75, -.4, 0.0, 1, 0, .0, .0] color:[0.4, 0.4, 0.4, 1]}
#_obs_bottom_right (table_base){  type:ssBox, size:[0.5 0.4 0.1 .005], contact:1 Q: [  -0.75, .8, 0.0, 1, 0, .0, .0] color:[0.4, 0.4, 0.4, 1]}
_obs_top_left (table_base){  type:ssBox, size:[0.5 1.2 0.1 .005], contact:1 Q: [  0.75, -.4, 0.0, 1, 0, .0, .0] color:[0.4, 0.4, 0.4, 1]}
#_obs_bottom_left (table_base){  type:ssBox, size:[0.5 0.4 0.1 .005], contact:1 Q: [  0.75, .8, 0.0, 1, 0, .0, .0] color:[0.4, 0.4, 0.4, 1]}
_obs_bottom_left (table_base){  type:ssBox, size:[2 0.4 0.1 .005], contact:1 Q: [  0., .8, 0.0, 1, 0, .0, .0] color:[0.4, 0.4, 0.4, 1]}
_obs_middle (table_base){  type:ssBox, size:[0.35 0.35 0.1 .005], contact:1 Q: [  0., .0, 0.0, 1, 0, .0, .0] color:[0.4, 0.4, 0.4, 1]}

_obs_right_bound (table_base){  type:ssBox, size:[0.05 2 0.1 .005], contact:1 Q: [  -0.975, .0, 0.0, 1, 0, .0, .0] color:[0.4, 0.4, 0.4, 1]}
_obs_left_bound (table_base){  type:ssBox, size:[0.05 2 0.1 .005], contact:1 Q: [  0.975, .0, 0.0, 1, 0, .0, .0] color:[0.4, 0.4, 0.4, 1]}
_obs_bottom_bound (table_base){  type:ssBox, size:[2 0.05 0.1 .005], contact:1 Q: [  0., 0.975, 0.0, 1, 0, .0, .0] color:[0.4, 0.4, 0.4, 1]}
_obs_top_bound (table_base){  type:ssBox, size:[2 0.05 0.1 .005], contact:1 Q: [  0., -0.975, 0.0, 1, 0, .0, .0] color:[0.4, 0.4, 0.4, 1]}

camera_init: { X: [0 0 3 0 1 0 0], width: 400 height: 400, focalLength: 1, zRange: [.5, 10]}
