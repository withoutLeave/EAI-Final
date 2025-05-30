import numpy as np

# we scale the depth value so that it can fit in np.uint16 and use image compression algorithms
DEPTH_IMG_SCALE = 16384

# simulation initialization settings

# TABLE_HEIGHT = 0.5
# # OBJ_INIT_TRANS = np.array([0.45, 0.2, 0.6])
# OBJ_INIT_TRANS = np.array([0.5,0.3,0.82])
# OBJ_RAND_RANGE = 0.3
# # OBJ_RAND_RANGE = 0.04
# OBJ_RAND_SCALE = 0.05

# # clip the point cloud to a box
# PC_MIN = np.array(
#     [
#         OBJ_INIT_TRANS[0] - OBJ_RAND_RANGE / 2,
#         OBJ_INIT_TRANS[1] - OBJ_RAND_RANGE / 2,
#         # 0.505,
#         0.6,
#     ]
# )
# PC_MAX = np.array(
#     [
#         OBJ_INIT_TRANS[0] + OBJ_RAND_RANGE / 2,
#         OBJ_INIT_TRANS[1] + OBJ_RAND_RANGE / 2,
#         # 0.65,
#         1.0,
#     ]
# )


TABLE_HEIGHT = 0.7
OBJ_INIT_TRANS = np.array([0.5, 0.3, 0.82])

OBJ_RAND_RANGE = 0.1
OBJ_RAND_SCALE = 0.02
PC_MIN = np.array(
    [
        OBJ_INIT_TRANS[0] - OBJ_RAND_RANGE / 2,
        OBJ_INIT_TRANS[1] - OBJ_RAND_RANGE / 2,
        0.75,
    ]
)

PC_MAX = np.array(
    [
        OBJ_INIT_TRANS[0] + OBJ_RAND_RANGE / 2,
        OBJ_INIT_TRANS[1] + OBJ_RAND_RANGE / 2,
        0.85,
    ]
)