"""Robot-specific initialization configurations for Metaworld environments.

This module defines joint initialization values and other robot-specific parameters
that need to be applied during environment reset for different robot arms.

To add a new robot:
1. Add the robot type to the ROBOT_JOINT_INIT_CONFIG dictionary
2. Specify the joint names and their initial qpos values
3. Add any other robot-specific initialization parameters
"""

from typing import Dict, List, Tuple, Optional

# Robot-specific joint initialization configuration
# Format: {robot_type: [(joint_name, qpos_value), ...]}
ROBOT_JOINT_INIT_CONFIG: Dict[str, List[Tuple[str, float]]] = {
    # Panda robot requires specific joint initialization
    "panda": [
        ("joint4", -2.5),
        ("joint6", 2.66),
        ("joint7", -1.6),
    ],
    
    # Sawyer robot (default) - no special joint initialization needed
    "sawyer": [],
    
    # UR5e robot - placeholder for future configuration
    "ur5e": [("elbow_joint", 1.6),
             ("shoulder_pan_joint", -2.5),],
    
    # UR10e robot - placeholder for future configuration
    "ur10e": [],
    
    # Kuka robot - placeholder for future configuration
    "kuka": [
    ],
    "xarm7": [
        ("joint2", -2.01),
        ],

    "gen3": [
       ("joint_1", -1.69671403),
        ("joint_2", 0.77121698),
        ("joint_3", 0.08691129),
        ("joint_4", 2.09581247),
        ("joint_5", 2.92317456),
        ("joint_6", -0.28289413),
        ("joint_7", -1.42432169)
        ],

    

    "unitree_z1": [
        ],

    
    # Unitree Z1 robot - placeholder for future configuration
    # To add configuration: uncomment and add joint initialization values
    # "unitree_z1": [
    #     ("joint_name", value),
    # ],
    
    # xArm7 robot - placeholder for future configuration
    # To add configuration: uncomment and add joint initialization values
    # "xarm7": [
    #     ("joint_name", value),
    # ],
}


def get_robot_joint_init_config(robot_type: Optional[str]) -> List[Tuple[str, float]]:
    """Get joint initialization configuration for a specific robot type.
    
    Args:
        robot_type: The robot type (e.g., "panda", "ur5e", "sawyer").
                   None or "unknown" will default to sawyer (no initialization).
    
    Returns:
        List of (joint_name, qpos_value) tuples for initialization.
        Returns an empty list if the robot type is not recognized or has no
        special initialization requirements.
    """
    # Default to sawyer (no special initialization) if robot_type is None or unknown
    if not robot_type or robot_type == "unknown":
        return ROBOT_JOINT_INIT_CONFIG.get("sawyer", [])
    
    # Get configuration for the specified robot type
    # Returns empty list if robot type not found
    return ROBOT_JOINT_INIT_CONFIG.get(robot_type, [])


def has_joint_init_config(robot_type: Optional[str]) -> bool:
    """Check if a robot type has joint initialization configuration.
    
    Args:
        robot_type: The robot type to check.
    
    Returns:
        True if the robot has joint initialization config, False otherwise.
    """
    config = get_robot_joint_init_config(robot_type)
    return len(config) > 0