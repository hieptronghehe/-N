import cv2
import numpy as np

def get_projection_matrix():
    """
    Return a homography matrix for camera projection.
    """
    # Interest points in camera (pixel)
    top_left_point = (345, 419)
    bottom_left_point = (126, 667)
    top_right_point = (693, 401)
    bottom_right_point = (817, 663)

    # Interest points in BIM coordinates (Project coordinates)
    top_left = (228, -203)
    bottom_left = (-201, -203)
    top_right = (230, 229)
    bottom_right = (-204, 209)

    # Get perspective transformation matrix
    pts1 = np.float32([top_left_point, bottom_left_point, top_right_point, bottom_right_point])
    pts2 = np.float32([top_left, bottom_left, top_right, bottom_right])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    return np.array(matrix)