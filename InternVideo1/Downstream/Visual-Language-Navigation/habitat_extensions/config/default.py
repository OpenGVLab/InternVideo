from typing import List, Optional, Union

from habitat.config.default import Config as CN
from habitat.config.default import get_config

_C = get_config()
_C.defrost()

# ----------------------------------------------------------------------------
# CUSTOM ACTION: HIGHTOLOWINFERENCE ACTION
# ----------------------------------------------------------------------------
_C.TASK.ACTIONS.HIGHTOLOWINFERENCE = CN()
_C.TASK.ACTIONS.HIGHTOLOWINFERENCE.TYPE = 'MoveHighToLowActionInference'
# ----------------------------------------------------------------------------
# CUSTOM ACTION: HIGHTOLOWEVAL ACTION
# ----------------------------------------------------------------------------
_C.TASK.ACTIONS.HIGHTOLOWEVAL = CN()
_C.TASK.ACTIONS.HIGHTOLOWEVAL.TYPE = 'MoveHighToLowActionEval'
# ----------------------------------------------------------------------------
# CUSTOM ACTION: HIGHTOLOW ACTION
# ----------------------------------------------------------------------------
_C.TASK.ACTIONS.HIGHTOLOW = CN()
_C.TASK.ACTIONS.HIGHTOLOW.TYPE = 'MoveHighToLowAction'
# ----------------------------------------------------------------------------
# GPS SENSOR
# ----------------------------------------------------------------------------
_C.TASK.GLOBAL_GPS_SENSOR = CN()
_C.TASK.GLOBAL_GPS_SENSOR.TYPE = "GlobalGPSSensor"
_C.TASK.GLOBAL_GPS_SENSOR.DIMENSIONALITY = 3
# ----------------------------------------------------------------------------
# # RXR INSTRUCTION SENSOR
# ----------------------------------------------------------------------------
_C.TASK.RXR_INSTRUCTION_SENSOR = CN()
_C.TASK.RXR_INSTRUCTION_SENSOR.TYPE = "RxRInstructionSensor"
_C.TASK.RXR_INSTRUCTION_SENSOR.features_path = "data/datasets/RxR_VLNCE_v0/text_features/rxr_{split}/{id:06}_{lang}_text_features.npz"
_C.TASK.RXR_INSTRUCTION_SENSOR.max_text_len = 512
_C.TASK.INSTRUCTION_SENSOR_UUID = "rxr_instruction"
# ----------------------------------------------------------------------------
# SHORTEST PATH SENSOR (previously: VLN_ORACLE_ACTION_SENSOR)
# ----------------------------------------------------------------------------
_C.TASK.SHORTEST_PATH_SENSOR = CN()
_C.TASK.SHORTEST_PATH_SENSOR.TYPE = "ShortestPathSensor"
# all goals can be navigated to within 0.5m.
_C.TASK.SHORTEST_PATH_SENSOR.GOAL_RADIUS = 0.5
# compatibility with the dataset generation oracle and paper results.
# if False, use the ShortestPathFollower in Habitat
_C.TASK.SHORTEST_PATH_SENSOR.USE_ORIGINAL_FOLLOWER = False
# ----------------------------------------------------------------------------
# VLN ORACLE PROGRESS SENSOR
# ----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR = CN()
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR.TYPE = "VLNOracleProgressSensor"
# ----------------------------------------------------------------------------
# NDTW MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.NDTW = CN()
_C.TASK.NDTW.TYPE = "NDTW"
_C.TASK.NDTW.SPLIT = "val_seen"
_C.TASK.NDTW.FDTW = True  # False: DTW
_C.TASK.NDTW.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json"
)
_C.TASK.NDTW.SUCCESS_DISTANCE = 3.0
# ----------------------------------------------------------------------------
# SDTW MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.SDTW = CN()
_C.TASK.SDTW.TYPE = "SDTW"
# ----------------------------------------------------------------------------
# PATH_LENGTH MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.PATH_LENGTH = CN()
_C.TASK.PATH_LENGTH.TYPE = "PathLength"
# ----------------------------------------------------------------------------
# ORACLE_NAVIGATION_ERROR MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.ORACLE_NAVIGATION_ERROR = CN()
_C.TASK.ORACLE_NAVIGATION_ERROR.TYPE = "OracleNavigationError"
# ----------------------------------------------------------------------------
# ORACLE_SUCCESS MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.ORACLE_SUCCESS = CN()
_C.TASK.ORACLE_SUCCESS.TYPE = "OracleSuccess"
_C.TASK.ORACLE_SUCCESS.SUCCESS_DISTANCE = 3.0
# ----------------------------------------------------------------------------
# ORACLE_SPL MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.ORACLE_SPL = CN()
_C.TASK.ORACLE_SPL.TYPE = "OracleSPL"
# ----------------------------------------------------------------------------
# STEPS_TAKEN MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.STEPS_TAKEN = CN()
_C.TASK.STEPS_TAKEN.TYPE = "StepsTaken"
# ----------------------------------------------------------------------------
# POSITION MEASUREMENT For faster eval
# ----------------------------------------------------------------------------
_C.TASK.POSITION = CN()
_C.TASK.POSITION.TYPE = 'Position'
# -----------------------------------------------------------------------------
# TOP_DOWN_MAP_VLNCE MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP_VLNCE = CN()
_C.TASK.TOP_DOWN_MAP_VLNCE.TYPE = "TopDownMapVLNCE"
_C.TASK.TOP_DOWN_MAP_VLNCE.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.TOP_DOWN_MAP_VLNCE.MAP_RESOLUTION = 1024
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_SOURCE_AND_TARGET = True
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_BORDER = True
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_SHORTEST_PATH = True
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_REFERENCE_PATH = True
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_FIXED_WAYPOINTS = True
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_MP3D_AGENT_PATH = True
_C.TASK.TOP_DOWN_MAP_VLNCE.GRAPHS_FILE = "data/connectivity_graphs.pkl"
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR = CN()
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR.DRAW = True
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR.FOV = 79
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR.VISIBILITY_DIST = 5.0
# ----------------------------------------------------------------------------
# DATASET EXTENSIONS
# ----------------------------------------------------------------------------
_C.DATASET.ROLES = ["guide"]  # options: "*", "guide", "follower"
# language options by region: "*", "te-IN", "hi-IN", "en-US", "en-IN"
_C.DATASET.LANGUAGES = ["*"]
# a list or set of episode IDs to allow in dataset creation. None allows all.
_C.DATASET.EPISODES_ALLOWED = None


def get_extended_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()

    if config_paths:
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)
    config.freeze()
    return config
