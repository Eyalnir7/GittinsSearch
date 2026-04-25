# Lgp Benchmarks


## keep in mind in g-files

* `logical: { is_gripper }` for the agent
* `logical: { is_place }` for tables (potentially goals)
* `logical: { is_pose }` for locations when the terminal is (poseEq goal obj)
* objects need to be children of table (to generate the (on table obj) as pre-condition of pick)
* objects need to have exact hight over table, so that poseConsistency at pick is ok (debug with -compute "[1 0]" -LGP/verbose 4 )
* make agents ssCylinder with size [.5 .1 .01] instead of cylinder (for differentiable collision)
* also make object with corner radius .01
