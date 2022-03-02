"""
Utils for preprocessing the Waymo dataset
"""
import math
import os
from typing import Tuple

import matplotlib.patches as mpathes
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# LaneCenter-Freeway = 1,
# LaneCenter-SurfaceStreet = 2,
# LaneCenter-BikeLane = 3,
# RoadLine-BrokenSingleWhite = 6,
# RoadLine-SolidSingleWhite = 7,
# RoadLine-SolidDoubleWhite = 8,
# RoadLine-BrokenSingleYellow = 9,
# RoadLine-BrokenDoubleYellow = 10,
# Roadline-SolidSingleYellow = 11,
# Roadline-SolidDoubleYellow=12,
# RoadLine-PassingDoubleYellow = 13,
# RoadEdgeBoundary = 15,
# RoadEdgeMedian = 16,
# StopSign = 17,
# Crosswalk = 18,
# SpeedBump = 19
ROAD_COLORS = {
    1: "grey",
    2: "grey",
    3: "grey",
    6: "white",
    7: "white",
    8: "white",
    9: "yellow",
    10: "yellow",
    11: "yellow",
    12: "yellow",
    13: "yellow",
    15: "green",
    16: "green",
    17: "red",
    18: "blue",
    19: "red",
}

# Unknown = 0,
# Arrow_Stop = 1,
# Arrow_Caution = 2,
# Arrow_Go = 3,
# Stop = 4,
# Caution = 5,
# Go = 6,
# Flashing_Stop = 7,
# Flashing_Caution = 8
TRAFFIC_COLORS = {
    0: "grey",
    1: "red",
    2: "yellow",
    3: "green",
    4: "red",
    5: "yellow",
    6: "green",
    7: "red",
    8: "yellow",
}


def build_features_description():
    """
    Set up the features description needed to use TFrecord dataset.

    Return:
        features_description: dict
        A dictionary of needed features.
    """
    roadgraph_features = {
        "roadgraph_samples/dir": tf.io.FixedLenFeature(
            [20000, 3], tf.float32, default_value=None
        ),
        "roadgraph_samples/id": tf.io.FixedLenFeature(
            [20000, 1], tf.int64, default_value=None
        ),
        "roadgraph_samples/type": tf.io.FixedLenFeature(
            [20000, 1], tf.int64, default_value=None
        ),
        "roadgraph_samples/valid": tf.io.FixedLenFeature(
            [20000, 1], tf.int64, default_value=None
        ),
        "roadgraph_samples/xyz": tf.io.FixedLenFeature(
            [20000, 3], tf.float32, default_value=None
        ),
    }

    scenario_features = {
        "scenario/id": tf.io.FixedLenFeature(
            [1], tf.string, default_value=None
        )
    }

    state_features = {
        "state/id": tf.io.FixedLenFeature(
            [128], tf.float32, default_value=None
        ),
        "state/type": tf.io.FixedLenFeature(
            [128], tf.float32, default_value=None
        ),
        "state/is_sdc": tf.io.FixedLenFeature(
            [128], tf.int64, default_value=None
        ),
        "state/tracks_to_predict": tf.io.FixedLenFeature(
            [128], tf.int64, default_value=None
        ),
        "state/current/bbox_yaw": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/height": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/length": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/timestamp_micros": tf.io.FixedLenFeature(
            [128, 1], tf.int64, default_value=None
        ),
        "state/current/valid": tf.io.FixedLenFeature(
            [128, 1], tf.int64, default_value=None
        ),
        "state/current/vel_yaw": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/velocity_x": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/velocity_y": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/width": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/x": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/y": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/z": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/future/bbox_yaw": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/height": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/length": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/timestamp_micros": tf.io.FixedLenFeature(
            [128, 80], tf.int64, default_value=None
        ),
        "state/future/valid": tf.io.FixedLenFeature(
            [128, 80], tf.int64, default_value=None
        ),
        "state/future/vel_yaw": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/velocity_x": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/velocity_y": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/width": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/x": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/y": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/z": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/past/bbox_yaw": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/height": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/length": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/timestamp_micros": tf.io.FixedLenFeature(
            [128, 10], tf.int64, default_value=None
        ),
        "state/past/valid": tf.io.FixedLenFeature(
            [128, 10], tf.int64, default_value=None
        ),
        "state/past/vel_yaw": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/velocity_x": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/velocity_y": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/width": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/x": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/y": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/z": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
    }

    traffic_light_features = {
        "traffic_light_state/current/state": tf.io.FixedLenFeature(
            [1, 16], tf.int64, default_value=None
        ),
        "traffic_light_state/current/valid": tf.io.FixedLenFeature(
            [1, 16], tf.int64, default_value=None
        ),
        "traffic_light_state/current/x": tf.io.FixedLenFeature(
            [1, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/current/y": tf.io.FixedLenFeature(
            [1, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/current/z": tf.io.FixedLenFeature(
            [1, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/past/state": tf.io.FixedLenFeature(
            [10, 16], tf.int64, default_value=None
        ),
        "traffic_light_state/past/valid": tf.io.FixedLenFeature(
            [10, 16], tf.int64, default_value=None
        ),
        "traffic_light_state/past/x": tf.io.FixedLenFeature(
            [10, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/past/y": tf.io.FixedLenFeature(
            [10, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/past/z": tf.io.FixedLenFeature(
            [10, 16], tf.float32, default_value=None
        ),
    }

    features_description = {}
    features_description.update(roadgraph_features)
    features_description.update(scenario_features)
    features_description.update(state_features)
    features_description.update(traffic_light_features)

    return features_description


def raw_data_parser(parsed: dict) -> list:
    """
    Parse the data loaded from TFRecord dataset into the ready to use style.

    Parameters:
    parsed: dict
        The dictionary containing all needed information.

    Return:
    out: List
        A list of needed information.
    """
    predict_mask = (parsed["state/tracks_to_predict"].numpy() == 1).squeeze()

    past_agents_locations = tf.stack(
        [parsed["state/past/x"], parsed["state/past/y"]], -1
    ).numpy()
    past_agents_sizes = tf.stack(
        [parsed["state/past/length"], parsed["state/past/width"]], -1
    ).numpy()
    past_agents_yaws = parsed["state/past/bbox_yaw"].numpy()
    past_agents_masks = parsed["state/past/valid"].numpy() == 1

    current_agents_location = (
        tf.stack([parsed["state/current/x"], parsed["state/current/y"]], -1)
        .numpy()
        .squeeze()
    )
    current_agents_size = (
        tf.stack(
            [parsed["state/current/length"], parsed["state/current/width"]], -1
        )
        .numpy()
        .squeeze()
    )
    current_agents_yaw = parsed["state/current/bbox_yaw"].numpy()
    current_agents_mask = (
        parsed["state/current/valid"].numpy() == 1
    ).squeeze()

    future_agents_locations = tf.stack(
        [parsed["state/future/x"], parsed["state/future/y"]], -1
    ).numpy()
    future_agents_sizes = tf.stack(
        [parsed["state/future/length"], parsed["state/future/width"]], -1
    ).numpy()
    future_agents_yaws = parsed["state/future/bbox_yaw"].numpy()
    future_agents_masks = parsed["state/future/valid"].numpy() == 1

    road_locations = parsed["roadgraph_samples/xyz"].numpy()[:, :2]
    road_types = parsed["roadgraph_samples/type"].numpy()
    road_valid = parsed["roadgraph_samples/valid"].numpy()
    road_ids = parsed["roadgraph_samples/id"].numpy()
    road_valid_masks = (road_valid == 1).squeeze()
    road_locations = road_locations[road_valid_masks]
    road_types = road_types[road_valid_masks]
    road_ids = road_ids[road_valid_masks]

    past_traffic_locations = tf.stack(
        [
            parsed["traffic_light_state/past/x"],
            parsed["traffic_light_state/past/y"],
        ],
        -1,
    ).numpy()
    past_traffic_states = parsed["traffic_light_state/past/state"].numpy()
    past_traffic_masks = parsed["traffic_light_state/past/valid"].numpy() == 1

    current_traffic_locations = tf.stack(
        [
            parsed["traffic_light_state/current/x"],
            parsed["traffic_light_state/current/y"],
        ],
        -1,
    ).numpy()
    current_traffic_states = parsed[
        "traffic_light_state/current/state"
    ].numpy()
    current_traffic_masks = (
        parsed["traffic_light_state/current/valid"].numpy() == 1
    )

    return [
        predict_mask,
        past_agents_locations,
        past_agents_sizes,
        past_agents_yaws,
        past_agents_masks,
        current_agents_location,
        current_agents_size,
        current_agents_yaw,
        current_agents_mask,
        future_agents_locations,
        future_agents_sizes,
        future_agents_yaws,
        future_agents_masks,
        road_locations,
        road_types,
        road_ids,
        past_traffic_locations,
        past_traffic_states,
        past_traffic_masks,
        current_traffic_locations,
        current_traffic_states,
        current_traffic_masks,
    ]


def render_img(parsed: dict, prefix: str) -> None:
    """
    Render the images.

    Parameters:
    parsed: dict
        The dictionary containing all needed information.
    prefix: str
        The path of folder to store images.

    Return:
    """
    (
        predict_mask,
        past_agents_locations,
        past_agents_sizes,
        past_agents_yaws,
        past_agents_masks,
        current_agents_location,
        current_agents_size,
        current_agents_yaw,
        current_agents_mask,
        _,
        _,
        _,
        _,
        road_locations,
        road_types,
        road_ids,
        past_traffic_locations,
        past_traffic_states,
        past_traffic_masks,
        current_traffic_locations,
        current_traffic_states,
        current_traffic_masks,
    ) = raw_data_parser(parsed)
    for index, mask in enumerate(predict_mask):
        if mask:
            path = os.path.join(prefix, "%d/" % index)
            if not os.path.exists(path):
                os.makedirs(path)
            # Setting up render parameters.
            target_location = current_agents_location[index]
            target_yaw = current_agents_yaw[index]
            project_location = [20, 40]
            project_width = 80
            project_height = 80
            min_x = -project_location[0]
            max_x = project_width - project_location[0]
            min_y = -project_location[1]
            max_y = project_height - project_location[1]
            meter_per_pixel = 0.2
            width = int(project_width / meter_per_pixel)
            height = int(project_height / meter_per_pixel)

            # Render road map.
            temp_road_locations = location_projection(
                target_location, target_yaw, road_locations
            )
            temp_road_types = road_types
            temp_road_ids = road_ids
            traffic_lines, road_lines = parse_road(
                temp_road_locations, temp_road_types, temp_road_ids
            )
            save_image(
                render_road_map(
                    width,
                    height,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    road_lines,
                    ROAD_COLORS,
                ),
                os.path.join(path, "road_map.png"),
            )

            # Render history information.
            for past_time in range(10):
                # Render traffic light.
                temp_traffic_location = past_traffic_locations[past_time, :, :]
                temp_traffic_states = past_traffic_states[past_time, :]
                temp_traffic_valid = past_traffic_masks[past_time]
                temp_traffic_location = temp_traffic_location[
                    temp_traffic_valid
                ]
                temp_traffic_states = temp_traffic_states[temp_traffic_valid]
                temp_traffic_location = location_projection(
                    target_location, target_yaw, temp_traffic_location
                )
                (traffic_light_lines, traffic_default_lines,) = parse_traffic(
                    traffic_lines,
                    temp_traffic_location,
                    temp_traffic_states,
                )
                save_image(
                    render_traffic(
                        width,
                        height,
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                        traffic_default_lines,
                        traffic_light_lines,
                        TRAFFIC_COLORS,
                    ),
                    os.path.join(path, "traffic_%d.png" % (10 - past_time)),
                )
                # Render agents.
                temp_agents_location = past_agents_locations[:, past_time, :]
                temp_agents_yaw = past_agents_yaws[:, past_time]
                temp_agents_size = past_agents_sizes[:, past_time]
                temp_agents_valid = past_agents_masks[:, past_time]
                temp_agents_location = temp_agents_location[temp_agents_valid]
                temp_agents_yaw = temp_agents_yaw[temp_agents_valid]
                temp_agents_size = temp_agents_size[temp_agents_valid]
                temp_agents_location = location_projection(
                    target_location,
                    target_yaw,
                    temp_agents_location,
                )
                temp_agents_yaw = yaw_projection(target_yaw, temp_agents_yaw)
                save_image(
                    render_bounding_box(
                        width,
                        height,
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                        temp_agents_location,
                        temp_agents_yaw,
                        temp_agents_size,
                    ),
                    os.path.join(path, "agents_%d.png" % (10 - past_time)),
                )
            # Render traffic.
            temp_traffic_location = current_traffic_locations
            temp_traffic_states = current_traffic_states
            temp_traffic_valid = current_traffic_masks
            temp_traffic_location = temp_traffic_location[temp_traffic_valid]
            temp_traffic_states = temp_traffic_states[temp_traffic_valid]
            temp_traffic_location = location_projection(
                target_location, target_yaw, temp_traffic_location
            )
            (traffic_light_lines, traffic_default_lines,) = parse_traffic(
                traffic_lines,
                temp_traffic_location,
                temp_traffic_states,
            )
            save_image(
                render_traffic(
                    width,
                    height,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    traffic_default_lines,
                    traffic_light_lines,
                    TRAFFIC_COLORS,
                ),
                os.path.join(path, "traffic.png"),
            )
            # Render agents
            temp_agents_location = current_agents_location
            temp_agents_yaw = current_agents_yaw
            temp_agents_size = current_agents_size
            temp_agents_valid = current_agents_mask
            temp_agents_location = temp_agents_location[temp_agents_valid]
            temp_agents_yaw = temp_agents_yaw[temp_agents_valid]
            temp_agents_size = temp_agents_size[temp_agents_valid]
            temp_agents_location = location_projection(
                target_location,
                target_yaw,
                temp_agents_location,
            )
            temp_agents_yaw = yaw_projection(target_yaw, temp_agents_yaw)
            save_image(
                render_bounding_box(
                    width,
                    height,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    temp_agents_location,
                    temp_agents_yaw,
                    temp_agents_size,
                ),
                os.path.join(path, "agents"),
            )


def location_projection(
    center: np.ndarray, yaw: float, data: np.ndarray
) -> np.ndarray:
    """
    Project the location data in global coordinate to the local coordinates
    around target vehicle. The target vehicle will lay at origin and
    head at positive x-axis direction.

    Parameters:
    center: np.ndarray
        the 2 dimensional global location of target vehicle
    yaw: float
        the yaw of target vehicle in rad
    data: np.ndarray
        the data to be projected

    Return:
    data: np.ndarray
        the projected data"""
    if len(data) > 0:
        direct = np.array([math.cos(yaw), math.sin(yaw)])
        orthogonal = np.array([-math.sin(yaw), math.cos(yaw)])
        data = data - center
        data = np.apply_along_axis(
            lambda x: [np.inner(x, direct), np.inner(x, orthogonal)], 1, data
        )
    return data


def yaw_projection(yaw: float, data: np.ndarray) -> np.ndarray:
    """
    Project the yaw data in global coordinate to the local coordinates
    around target vehicle. The target vehicle will head at positive
    x-axis direction.

    Parameters:
    yaw: float
        the yaw of target vehicle in rad
    data: np.ndarray
        the data to be projected

    Return:
    data: np.ndarray
        the projected data"""
    data = data - yaw
    return data


def render_traffic(
    width: int,
    height: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    traffic_default_lines: list,
    traffic_light_lines: list,
    traffic_colors: dict,
) -> np.ndarray:
    """
    Render road map.

    Parameters:
    width: int
        The width in pixel of the rendered fig.
    height: int
        The height in pixel of the rendered fig.
    min_x: float
        The minimal x coordinate in meters to be rendered.
    max_x: float
        The maximal x coordinate in meters to be rendered.
    min_y: float
        The minimal y coordinate in meters to be rendered.
    max_y: float
        The minimal y coordinate in meters to be rendered.
    traffic_default_lines: list
        The central lines which are not controlled by traffic lights
        to be rendered. List of shape (N,2) where
        N is the number of lines, the first term of each slice is
        sample points along the line and the second term is the type
        of the line.
    traffic_light_lines: list
        The central lines which are controlled by traffic lights
        to be rendered. List of shape (N,2) where
        N is the number of lines, the first term of each slice is
        sample points along the line and the second term is the traffic
        light state associated with the line.
    traffic_colors: dict
        The dictionary of colors to render different types of central
        lines. Keys are the types of road and values are the colors to
        be used.

    Return:
    data_traffic: np.ndarray
        An ndarray of size (height, width, 3) which is the rgb value of
        rendered fig."""
    figure, axes = plt.subplots(1, 1)
    dpi = 100
    figure.set_size_inches([width / dpi, height / dpi])
    figure.set_dpi(dpi)
    figure.set_facecolor("black")
    axes.set_axis_off()
    axes.set_xlim([min_x, max_x])
    axes.set_ylim([min_y, max_y])
    figure.tight_layout(pad=0)
    axes.grid(False)
    for traffic_light_line, traffic_light_state in traffic_default_lines:
        axes.plot(
            traffic_light_line[:, 0],
            traffic_light_line[:, 1],
            color=traffic_colors[traffic_light_state],
            alpha=1,
            ms=2,
        )
    for traffic_light_line, traffic_light_state in traffic_light_lines:
        axes.plot(
            traffic_light_line[:, 0],
            traffic_light_line[:, 1],
            color=traffic_colors[traffic_light_state],
            alpha=1,
            ms=2,
        )
    figure.canvas.draw()
    data_traffic = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    data_traffic = data_traffic.reshape(
        figure.canvas.get_width_height()[::-1] + (3,)
    )
    plt.close()
    return data_traffic


def render_road_map(
    width: int,
    height: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    road_lines: list,
    road_colors: dict,
) -> np.ndarray:
    """
    Render road map.

    Parameters:
    width: int
        The width in pixel of the rendered fig.
    height: int
        The height in pixel of the rendered fig.
    min_x: float
        The minimal x coordinate in meters to be rendered.
    max_x: float
        The maximal x coordinate in meters to be rendered.
    min_y: float
        The minimal y coordinate in meters to be rendered.
    max_y: float
        The minimal y coordinate in meters to be rendered.
    road_lines: list
        The roads to be rendered. List of shape (N,2) where
        N is the number of roads, the first term of each slice is
        sample points along the road and the second term is the type
        of the road.
    road_colors: dict
        The dictionary of colors to render different types of road. Keys
        are the types of road and values are the colors to be used.

    Return:
    data_road: np.ndarray
        An ndarray of size (height, width, 3) which is the rgb value of
        rendered fig."""
    figure, axes = plt.subplots(1, 1)
    dpi = 100
    figure.set_size_inches([width / dpi, height / dpi])
    figure.set_dpi(dpi)
    figure.set_facecolor("black")
    axes.set_axis_off()
    axes.set_xlim([min_x, max_x])
    axes.set_ylim([min_y, max_y])
    figure.tight_layout(pad=0)
    axes.grid(False)
    for road_line, road_type in road_lines:
        axes.plot(
            road_line[:, 0],
            road_line[:, 1],
            color=road_colors[road_type],
            alpha=1,
            ms=2,
        )
    figure.canvas.draw()
    data_road = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    data_road = data_road.reshape(
        figure.canvas.get_width_height()[::-1] + (3,)
    )
    plt.close()
    return data_road


def render_bounding_box(
    width: int,
    height: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    agents_location: np.ndarray,
    agents_yaw: np.ndarray,
    agents_size: np.ndarray,
) -> np.ndarray:
    """
    Render road map.

    Parameters:
    width: int
        The width in pixel of the rendered fig.
    height: int
        The height in pixel of the rendered fig.
    min_x: float
        The minimal x coordinate in meters to be rendered.
    max_x: float
        The maximal x coordinate in meters to be rendered.
    min_y: float
        The minimal y coordinate in meters to be rendered.
    max_y: float
        The minimal y coordinate in meters to be rendered.
    agents_location: np.ndarray
        The locations in meters of agents to be rendered.
    agents_yaw: np.ndarray
        The yaw in rad of agents to be rendered.
    agents_size: np.ndarray
        The size in meters of agents to be rendered.

    Return:
    data: np.ndarray
        An ndarray of size (height, width, 3) which is the rgb value of
        rendered fig."""
    figure, axes = plt.subplots(1, 1)
    dpi = 100
    figure.set_size_inches([width / dpi, height / dpi])
    figure.set_dpi(dpi)
    figure.set_facecolor("black")
    axes.set_axis_off()
    axes.set_xlim([min_x, max_x])
    axes.set_ylim([min_y, max_y])
    figure.tight_layout(pad=0)
    axes.grid(False)
    for location, yaw, size in zip(agents_location, agents_yaw, agents_size):
        agent_length, agent_width = size
        direct = np.array([math.cos(yaw), math.sin(yaw)])
        orthogonal = np.array([-math.sin(yaw), math.cos(yaw)])
        rect = mpathes.Rectangle(
            location
            - agent_length / 2 * direct
            - agent_width / 2 * orthogonal,
            agent_length,
            agent_width,
            float(np.degrees(yaw)),
            color="w",
        )
        axes.add_patch(rect)
    figure.canvas.draw()
    data = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def parse_traffic(
    traffic_lines: list,
    traffic_location: np.ndarray,
    traffic_states: np.ndarray,
) -> Tuple[list, list]:
    """
    Parse traffic lines into those controlled by traffic lights and
    those not.

    Parameters:
    traffic_lines: list,
        List of central lines.
    traffic_location: np.ndarray,
        (N,2) ndarray of locations of traffic lights.
    traffic_location: np.ndarray,
        (N,1) ndarray of states of traffic lights.

    Return:
    traffic_light_lines: list
        List of traffic lines which are controlled by traffic lights.
    traffic_default_lines: list
        List of traffic lines which are not controlled by traffic lights.
    """
    traffic_light_lines = []
    traffic_default_lines = []
    for traffic_line, _ in traffic_lines:
        controlled_flag = False
        for traffic_light, traffic_state in zip(
            traffic_location, traffic_states
        ):
            min_distance = (
                ((traffic_line - traffic_light) ** 2).sum(axis=1).min()
            )
            if min_distance < 1:
                traffic_light_lines.append([traffic_line, traffic_state])
                controlled_flag = True
                break
        if not controlled_flag:
            traffic_default_lines.append([traffic_line, 0])
    return traffic_light_lines, traffic_default_lines


def parse_road(
    road_locations: np.ndarray, road_types: np.ndarray, road_ids: np.ndarray
) -> Tuple[list, list]:
    """
    Parse road map data into road lines and central lines.

    Parameters:
    road_location: np.ndarray,
        (N,2) ndarray of sample points along the roads.
    road_type: np.ndarray,
        (N,1) ndarray of types of sample points.
    road_id: np.ndarray,
        (N,1) ndarray of id of the feature maps the sample points belong
        to.

    Return:
    traffic_lines: list
        List of central lines.
    road_lines: list
        List of road lines.
    """
    traffic_lines = []
    road_lines = []
    for feature_id in np.unique(road_ids):
        id_mask = (road_ids == feature_id).squeeze()
        id_road = road_locations[id_mask]
        id_type = road_types[id_mask]
        assert np.unique(id_type).size == 1
        road_type = id_type[0][0]
        if road_type not in ROAD_COLORS:
            continue
        if road_type in {1, 2, 3}:
            traffic_lines.append([id_road, road_type])
        elif road_type in (18, 19):
            id_road = np.append(id_road, [id_road[0]], axis=0)
            road_lines.append([id_road, road_type])
        else:
            road_lines.append([id_road, road_type])
    return traffic_lines, road_lines


def save_image(data, filename):
    """
    Save rgb image data as an image to filename.
    """
    sizes = np.shape(data)
    fig = plt.figure()
    fig.set_size_inches(1.0 * sizes[1] / sizes[0], 1, forward=False)
    axes = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    axes.set_axis_off()
    fig.add_axes(axes)
    axes.imshow(data)
    plt.savefig(filename, dpi=sizes[0], cmap="hot")
    plt.close()
