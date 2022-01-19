#!/usr/bin/env python2
import os
import sys

import rospy
from tf.transformations import quaternion_from_euler
from actionlib import SimpleActionClient

# import ros messages
from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Quaternion

class BWIbot(object):
    def __init__(self, entity_name):
        self.entity_name = entity_name

        rospy.loginfo("Connecting to {}...".format(os.path.join(self.entity_name,'move_base')))
        self.move_base = SimpleActionClient( os.path.join(self.entity_name, "move_base"), MoveBaseAction)
        connected = self.move_base.wait_for_server(timeout=rospy.Duration(30.0))
        if not connected:
            rospy.logerr("{}/move_base does not respond within 30 seconds")
            sys.exit()

        # define parameters with default value
        self.done = False
        self.traj = []

    def move(self, goal_x, goal_y, goal_yaw, record=False):
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp       = rospy.Time.now()
        goal.target_pose.header.frame_id    = os.path.join(self.entity_name, 'level_mux_map')
        goal.target_pose.pose.position = Point(goal_x, goal_y, 0.0)
        goal.target_pose.pose.orientation = Quaternion( *quaternion_from_euler(0,0,goal_yaw) )

        if record is True:
            self.move_base.send_goal(goal, active_cb=self.begin_cb, done_cb=self.arrive_cb, feedback_cb = self.log_trajectory)
        else:
            self.move_base.send_goal(goal, active_cb=self.begin_cb, done_cb=self.arrive_cb)

    # Define callback functions
    def begin_cb(self):
        self.done = False
        self.traj = []
        self.start = rospy.Time.now()

    def arrive_cb(self, state, result):
        self.done = True
        self.ttd = (rospy.Time.now() - self.start).to_sec()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo("{} succeeded".format(self.entity_name))
        elif state == GoalStatus.ABORTED:
            rospy.loginfo("{} aborted".format(self.entity_name))

    def log_trajectory(self, feedback):
        print(feedback)
        self.traj.append(feedback)

if __name__ == "__main__":
    rospy.init_node("rl_train_episode_py")
    rospy.sleep(1.0)

    marvin = BWIbot(entity_name="marvin")

    marvin.move(0, 4, -1.57, record=True)
    while not marvin.done:
        rospy.sleep(10.0)

    print("Time to destination: {:.1f}s".format(marvin.ttd))
    print("Number of recorded trajectory: {}".format(len(marvin.traj)))
