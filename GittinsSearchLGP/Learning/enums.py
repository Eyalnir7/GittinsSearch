import enum

class TaskType(enum.Enum):
    FEASIBILITY = 1
    QUANTILE_REGRESSION_FEAS = 2
    QUANTILE_REGRESSION_INFEAS = 3

class NodeType(enum.Enum):
    WAYPOINTS = 1
    RRT = 2
    LGP = 3