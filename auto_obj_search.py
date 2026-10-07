# Author: Yared Fente, CO'29
# Last Edit: 21 July 2026

""" Use AutoWalk to deploy Spot on missions to search and locate objects. """

import argparse
import logging
import math
import os
import sys
import time
import traceback
import cv2
import numpy as np

import google.protobuf.timestamp_pb2
import graph_nav_util
import grpc

import bosdyn.client.channel
import bosdyn.client.util
from bosdyn.api import geometry_pb2, power_pb2, robot_state_pb2, image_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient, power_on_motors, safe_power_off_motors
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient


from ultralytics import YOLO
from PIL import Image

# Set of supported objects.
OBJECTS = {"backpack", "bottle", "chair", 
           "cup", "laptop", "book", 
           "ball", "box", "bag"}

class UnknownObjError(Exception):
    pass

class AutoObjSearch(object):

    def __init__(self, robot, upload_path: str, confidence_threshold: float = 0.65):

        self._robot = robot
        self._upload_path = upload_path
        self._confidence_threshold = confidence_threshold

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Create robot state and command clients.
        self._robot_command_client = self._robot.ensure_client(
            RobotCommandClient.default_service_name)
        self._robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)

        # Create the client for the Graph Nav main service.
        self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)

        # Create a power client for the robot.
        self._power_client = self._robot.ensure_client(PowerClient.default_service_name)

        # Create an image client for the robot.
        self._image_client = self._robot.ensure_client(ImageClient.default_service_name)

        # Create a lease client for the robot.
        self._lease_client = self._robot.ensure_client(LeaseClient.default_service_name)

        self._source_names = [
            src.name for src in self._image_client.list_image_sources() if
            (src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL and 'depth' not in src.name)
        ]

        # Boolean indicating the robot's power state.
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        self._powered_on = self._started_powered_on

        # Loading the model
        self._model = YOLO("yolo26s.pt")

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()

        
        self._target_color = None
        self._target_object = None

        self._found = False

        # Filepath for uploading a saved graph's and snapshots too.
        if upload_path[-1] == '/':
            self._upload_path = upload_path[:-1]
        else:
            self._upload_path = upload_path

    def _get_localization_state(self):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state(request_gps_state=self.use_gps)
        print(f'Got localization: \n{state.localization}')
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        print(f'Got robot state in kinematic odometry frame: \n{odom_tform_body}')
        if self.use_gps:
            print(f'GPS info:\n{state.gps}')

    def _set_initial_localization_fiducial(self):
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(initial_guess_localization=localization,
                                                ko_tform_body=current_odom_tform_body)

    def _clear_graph_and_cache(self):
        """Clear the state of the map on the robot, removing all waypoints and
        edges.

        Also clears the disk cache.
        """
        return self._graph_nav_client.clear_graph_and_cache()

    # @do_not_publish_end

    def _list_graph_waypoint_and_edge_ids(self):
        """List the waypoint ids and edge ids of the graph currently on the
        robot."""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Empty graph.')
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = graph_nav_util.update_waypoints_and_edges(
            graph, localization_id)

    def _upload_graph_and_snapshots(self):
        """Upload the graph and snapshots to the robot."""
        print('Loading the graph from disk into local storage...')
        with open(self._upload_filepath + '/graph', 'rb') as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            print(
                f'Loaded graph has {len(self._current_graph.waypoints)} waypoints and {len(self._current_graph.edges)} edges'
            )
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(f'{self._upload_filepath}/waypoint_snapshots/{waypoint.snapshot_id}',
                      'rb') as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(f'{self._upload_filepath}/edge_snapshots/{edge.snapshot_id}',
                      'rb') as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        print('Uploading the graph and snapshots to the robot...')
        time_before = time.time()
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(graph=self._current_graph,
                                                       generate_new_anchoring=true_if_empty)
        # Upload any missing snapshots to the robot.
        upload_individually = False
        try:
            self._graph_nav_client.upload_snapshots(
                graph_nav_pb2.UploadSnapshotsRequest.Snapshots(waypoint_snapshots=[],
                                                               edge_snapshots=[]))
        except:
            # An empty UploadSnapshots request failed, fall back to slow RPC.
            upload_individually = True

        if upload_individually:
            for snapshot_id in response.unknown_waypoint_snapshot_ids:
                waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
                self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
                print(f'Uploaded {waypoint_snapshot.id}')
            for snapshot_id in response.unknown_edge_snapshot_ids:
                edge_snapshot = self._current_edge_snapshots[snapshot_id]
                self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
                print(f'Uploaded {edge_snapshot.id}')
        else:
            # Upload in groups of 16MB.
            kMaxBytes = 16 * 1024 * 1024
            snapshots = []
            num_bytes = 0

            # Upload waypoint snapshots.
            for snapshot_id in response.unknown_waypoint_snapshot_ids:
                this_bytes = self._current_waypoint_snapshots[snapshot_id].ByteSize()
                if len(snapshots) > 0 and this_bytes + num_bytes > kMaxBytes:
                    print(f'Uploading {len(snapshots)} waypoint snapshots')
                    self._graph_nav_client.upload_snapshots(
                        graph_nav_pb2.UploadSnapshotsRequest.Snapshots(
                            waypoint_snapshots=snapshots, edge_snapshots=[]))
                    snapshots = []
                    num_bytes = 0
                snapshots.append(self._current_waypoint_snapshots[snapshot_id])
                num_bytes += this_bytes
            if len(snapshots) > 0:
                print(f'Uploading final {len(snapshots)} waypoint snapshots')
                self._graph_nav_client.upload_snapshots(
                    graph_nav_pb2.UploadSnapshotsRequest.Snapshots(waypoint_snapshots=snapshots,
                                                                   edge_snapshots=[]))

            # Upload edge snapshots.
            snapshots = []
            num_bytes = 0
            for snapshot_id in response.unknown_edge_snapshot_ids:
                this_bytes = self._current_edge_snapshots[snapshot_id].ByteSize()
                if len(snapshots) > 0 and this_bytes + num_bytes > kMaxBytes:
                    print(f'Uploading {len(snapshots)} edge snapshots')
                    self._graph_nav_client.upload_snapshots(
                        graph_nav_pb2.UploadSnapshotsRequest.Snapshots(
                            waypoint_snapshots=[], edge_snapshots=snapshots))
                    snapshots = []
                    num_bytes = 0
                snapshots.append(self._current_edge_snapshots[snapshot_id])
                num_bytes += this_bytes
            if len(snapshots) > 0:
                print(f'Uploading final {len(snapshots)} edge snapshots')
                self._graph_nav_client.upload_snapshots(
                    graph_nav_pb2.UploadSnapshotsRequest.Snapshots(waypoint_snapshots=[],
                                                                   edge_snapshots=snapshots))
        upload_time = time.time() - time_before
        print(
            f'Uploaded graph and {len(response.unknown_waypoint_snapshot_ids)} (of {len(self._current_graph.waypoints)}) waypoints and {len(response.unknown_edge_snapshot_ids)} (of {len(self._current_graph.edges)}) edges, elapsed time {round(upload_time * 1000)}ms'
        )

        # The upload is complete! Check that the robot is localized to the graph,
        # and if it is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print('\n')
            print(
                'Upload complete! The robot is currently not localized to the map; please localize the robot.')

    def _navigate_to(self, *args):
        """Navigate to a specific waypoint."""
        # Take the first argument as the destination waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            print('No waypoint provided as a destination for navigate to.')
            return

        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            print('Failed to power on the robot, and cannot complete navigate to request.')
            return

        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   command_id=nav_to_cmd_id)
            except ResponseError as e:
                print(f'Error while navigating {e}')
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        # Power off the robot if appropriate.
        if self._powered_on and not self._started_powered_on:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)

    def _clear_graph(self):
        """Clear the state of the map on the robot, removing all waypoints and
        edges."""
        return self._graph_nav_client.clear_graph()

    def toggle_power(self, should_power_on):
        """Power the robot on/off dependent on the current power state."""
        is_powered_on = self.check_is_powered_on()
        if not is_powered_on and should_power_on:
            # Power on the robot up before navigating when it is in a powered-off state.
            power_on_motors(self._power_client)
            motors_on = False
            while not motors_on:
                future = self._robot_state_client.get_robot_state_async()
                state_response = future.result(
                    timeout=10)  # 10 second timeout for waiting for the state response.
                if state_response.power_state.motor_power_state == robot_state_pb2.PowerState.STATE_ON:
                    motors_on = True
                else:
                    # Motors are not yet fully powered on.
                    time.sleep(.25)
        elif is_powered_on and not should_power_on:
            # Safe power off (robot will sit then power down) when it is in a
            # powered-on state.
            safe_power_off_motors(self._robot_command_client, self._robot_state_client)
        else:
            # Return the current power state without change.
            return is_powered_on
        # Update the locally stored power state.
        self.check_is_powered_on()
        return self._powered_on

    def check_is_powered_on(self):
        """Determine if the robot is powered on or off."""
        power_state = self._robot_state_client.get_robot_state().power_state
        self._powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        return self._powered_on

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit
        when command succeeds."""
        if command_id == -1:
            # No command, so we have no status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print('Robot got lost when navigating the route, the robot will now sit down.')
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print('Robot got stuck when navigating the route, the robot will now sit down.')
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print('Robot is impaired.')
            return True
        else:
            # Navigation command is not complete yet.
            return False

    def _on_quit(self):
        """Cleanup on quit from the command line interface."""
        # Sit the robot down + power off after the navigation command is complete.
        if self._powered_on and not self._started_powered_on:
            self._robot_command_client.robot_command(RobotCommandBuilder.safe_power_off_command(),
                                                     end_time_secs=time.time())

    def _capture_images(self):
        """Retrieve images from every visual camera source using the image client."""
        image_responses = self._image_client.get_image_from_sources(self._source_names)

        return [
            {
                'image_response': image_response,
                'image': self._decode_image(image_response),
                'source': image_response.source.name
            }
            for image_response in image_responses
        ]
    
    def _decode_image(self, image_response):
        """Decodes the image captured by Spot's camera and returns a NumPy array."""
        image = image_response.shot.image

        if image.format == image_pb2.Image.FORMAT_JPEG:
            arr = np.frombuffer(image.data, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR) # May return none, need checks later

        else:
            raise ValueError(f'Unsupported image format.')
        
        return decoded
    
    def _detect_objects(self, image):
        """Conducts inference on the images captured by Spot and returns a list of dictionaries detailing the results."""
        results = self._model.predict(
            source=image,
            conf=self._confidence_threshold,
            verbose=False
        )

        result = results[0]
        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            detection = {}

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection["class_id"] = class_id
            detection["class_name"] = result.names[class_id]
            detection["confidence"] = confidence
            detection["center_xy"] = (
                (x1 + x2) / 2,
                (y1 + y2) / 2
            )
            detection["bbox"] = (x1, y1, x2, y2)

            detections.append(detection)

        return detections

    def _analyze_images(self):
        """Captures images from Spot and runs inference on every image."""
        captured_images = self._capture_images()

        for captured_image in captured_images:

            captured_image["detections"] = self._detect_objects(captured_image["image"])

        return captured_images

    def _filter_detections(self, analyzed_images: list, target_object: str):
        """Filters out detections of objects that aren't the target."""

        filtered_detections = []

        for analyzed_image in analyzed_images:

            for detection in analyzed_image["detections"]:
                if detection["class_name"] == target_object:

                    filtered_detections.append({
                        "source": analyzed_image["source"],
                        "image_response": analyzed_image["image_response"],
                        "image": analyzed_image["image"],
                        "detection": detection
                    })

        return filtered_detections

    # def _estimate_target_position(self):
    #     """Uses the center pixel of the target object within an image to determine position relative to Spot."""

    def _search_for_target(self, target_object: str):
        """Combines the functionality of _analyze_images & _filter_detections."""
        analyzed_images = self._analyze_images()

        target_detections = self._filter_detections(analyzed_images, target_object)

        if not target_detections:
            return None

        return max(
            target_detections,
            key=lambda target: target["detection"]["confidence"]
        )

    def setup(self):
        """Prepare GraphNav client."""
        self._clear_graph_and_cache()
        self._upload_graph_and_snapshots()
        self._list_graph_waypoint_and_edge_ids()
        self._set_initial_localization_fiducial()

    def run(self, target_object: str):
        """"""
        for count, waypoint in enumerate(self._current_graph.waypoints):

            self._navigate_to(waypoint.id)

            target_detection = self._search_for_target(target_object)

            if target_detection is not None:
                print(
                    f"Found {target_object} at waypoint #{count} with ID: {waypoint.id} using {target_detection['source']}."
                )
                return target_detection

        print(f"Could not find {target_object}.")
        return None

def main():
    """Command Line Interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--upload-filepath', 
                        help='Full filepath to graph and snapshots to be uploaded.', required=True)
    parser.add_argument('-conf', '--confidence-threshold',
                        help='Minimum confidence score (0.0-1.0) required for a YOLO detection to be accepted', required=False, default=0.65)
    parser.add_argument('-t', '--target-object',
                        help='Name of the object class Spot is searching for.', required=True)
    bosdyn.client.util.add_base_arguments(parser)   
    options = parser.parse_args()

    if not 0.0 <= options.confidence_threshold <= 1.0:
        parser.error("Confidence threshold must be between 0.0 and 1.0")

    # Setup and authenticate the robot.
    sdk = bosdyn.client.create_standard_sdk('AutoObjSearchClient')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    auto_obj_search = AutoObjSearch(robot, upload_path=options.upload_filepath, confidence_threshold=options.confidence_threshold)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                auto_obj_search.setup()
                auto_obj_search.run(options.target_object)
                return True
            except Exception as exc:  # pylint: disable=broad-except
                print(exc)
                print('AutoObjSearch encountered an error.')
                return False
    except ResourceAlreadyClaimedError:
        print(
            'The robot\'s lease is currently in use. Check for a tablet connection or try again in a few seconds.'
        )
        return False

if __name__ == '__main__':
    if not main():
        sys.exit(1)