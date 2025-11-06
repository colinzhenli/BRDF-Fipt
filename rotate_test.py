import numpy as np

def rotate_pose_around_axis(pose, axis, angle_degrees):
    """
    Rotate a 4x4 pose matrix around a given 3D axis.

    Args:
        pose (np.ndarray): 4x4 transformation matrix.
        axis (np.ndarray): 3D rotation axis.
        angle_degrees (float): rotation angle in degrees.

    Returns:
        np.ndarray: rotated 4x4 pose matrix.
    """
    assert pose.shape == (4, 4)
    assert axis.shape == (3,)

    # Normalize the axis
    axis = axis / np.linalg.norm(axis)

    # Convert to radians
    theta = np.deg2rad(angle_degrees)

    # Rodrigues’ rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R_axis = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    # Current rotation and translation
    R = pose[:3, :3]
    t = pose[:3, 3]

    # Rotate the rotation and translation part
    R_new = R_axis @ R
    t_new = R_axis @ t

    # Construct new pose
    pose_rotated = np.eye(4)
    pose_rotated[:3, :3] = R_new
    pose_rotated[:3, 3] = t_new

    return pose_rotated


if __name__ == "__main__":
    # Example
    pose = np.eye(4)
    pose = np.eye(4)
    pose[:3, :3] = np.array([
        [ 0.028322419006849342, -0.9728096963848317,  -0.22986764713906993],
        [-0.006689541023191124,  0.22977028798813612, -0.9732218990542435],
        [ 0.9995764556163274,     0.029101707467097147, 0.0]
    ])
    pose[:3, 3] = np.array([-166.2422773567819, 32.80531565136983, 11.698886401773052])
    axis = np.array([0, 0, 1])  # rotate around Z-axis
    angle = -143.3741650246211  # degrees

    rotated_pose = rotate_pose_around_axis(pose, axis, angle)
    print("Original pose:\n", pose)
    print("Rotated pose:\n", rotated_pose)
