#!/usr/bin/env python3
"""
Exploration episode; Each actions are randomly chosen.
"""
# Load basic modules
import os
import sys
import uuid
import copy
import torch
import numpy as np
import pickle
import argparse
# Load ros related modules
import rospy
import rospkg
from actionlib import SimpleActionClient
# Load ros messages
from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Point, Quaternion
from lidar_hallucination.msg import VirtualCircle, VirtualCircles
# Load custom TD3 network
import TD3

def quaternion_from_euler( r, p, y):
    qx = np.sin(r/2) * np.cos(p/2) * np.cos(y/2) - np.cos(r/2) * np.sin(p/2) * np.sin(y/2)
    qy = np.cos(r/2) * np.sin(p/2) * np.cos(y/2) + np.sin(r/2) * np.cos(p/2) * np.sin(y/2)
    qz = np.cos(r/2) * np.cos(p/2) * np.sin(y/2) - np.sin(r/2) * np.sin(p/2) * np.cos(y/2)
    qw = np.cos(r/2) * np.cos(p/2) * np.cos(y/2) + np.sin(r/2) * np.sin(p/2) * np.sin(y/2)
    return [qx, qy, qz, qw]

class BWIbot(object):
    def __init__(self, entity_name, random_episode=False,
                 max_action = 10.0, expl_noise = 0.01, action_dim = 6):
        self.entity_name = entity_name
        self.random_episode = random_episode
        self.max_action = max_action
        self.expl_noise = expl_noise
        self.action_dim = action_dim

        rospy.loginfo("Connecting to {}...".format( os.path.join(self.entity_name, "move_base") ))
        self.move_base = SimpleActionClient( os.path.join(self.entity_name, "move_base"), MoveBaseAction )
        connected = self.move_base.wait_for_server( timeout = rospy.Duration(30.0) )
        if not connected:
            rospy.logerr("{}/move_base does not respond within 30 seconds. Aborting...".format(self.entity_name))
            sys.exit()

        # Load frozen network
        self.policy = TD3.TD3()
        pkg_dir     = rospkg.RosPack().get_path("lidar_hallucination")
        network_dir = os.path.join(pkg_dir, "model", "td3")
        self.data_storage = os.path.join(pkg_dir, "result")
        self.policy.load(network_dir)

        # define state generation parameters
        self.resolution = 0.1
        self.radius = 0.24
        self.rpx = np.ceil(self.radius/self.resolution).astype(np.int)
        grid_x, grid_y = (np.mgrid[0:2*self.rpx+1, 0:2*self.rpx+1] - self.rpx) * self.resolution
        dist = (grid_x)**2 + (grid_y)**2
        self.mask = (dist <= self.radius**2).astype(np.float32)

        self.scan_msg = rospy.wait_for_message(os.path.join(entity_name, "scan_filtered"), LaserScan)
        self.theta = np.arange(self.scan_msg.angle_min,
                               self.scan_msg.angle_max + self.scan_msg.angle_increment,
                               self.scan_msg.angle_increment)
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = os.path.join(self.entity_name, "level_mux_map")

        # Define publisher and subscriber
        self.pub_action = rospy.Publisher(os.path.join(entity_name, "add_circle"), VirtualCircles, queue_size=10)
        self._sub_scan = rospy.Subscriber(os.path.join(entity_name, "scan_filtered"), LaserScan, self.scan_cb)

        # Define common parameters
        self.finish, self.ttd, self.trajectory = False, None, []
        self.prev_location = None
        self.states, self.actions, self.rewards, self.done = [], [], [], []

    def move(self, x, y, yaw):
        self.goal.target_pose.header.frame_id  = os.path.join(self.entity_name, 'level_mux_map')
        self.goal.target_pose.pose.position    = Point(x, y, 0.0)
        self.goal.target_pose.pose.orientation = Quaternion( *quaternion_from_euler(0.0, 0.0, yaw) )

        self.move_base.send_goal(self.goal, active_cb   = self.start_cb,
                                            feedback_cb = self.periodic_cb, 
                                            done_cb     = self.finish_cb)

    def get_state(self, scan_msg):
        state_img = np.zeros((128+2*self.rpx, 128+2*self.rpx), dtype=np.float32)
        valid_idx = np.isfinite( scan_msg.ranges )
        r  = np.array( scan_msg.ranges )[valid_idx]
        th = self.theta[valid_idx]

        Xs = r * np.cos(th)
        Ys = r * np.sin(th)

        # only leaves valid Xs and Ys
        #  0.0m < PXs < 12.8m
        # -6.4m < PYs < +6.4m
        valid_idx = np.logical_and(Xs < 12.8, np.abs(Ys) < 6.4)
        PXs = (Xs[valid_idx] / self.resolution).astype(np.int) + (self.rpx)
        PYs = (Ys[valid_idx] / self.resolution).astype(np.int) + (self.rpx + 64)

        # Make circula mask
        for x,y in zip(PXs, PYs):
            state_img[x-self.rpx:x+self.rpx + 1, y-self.rpx:y+self.rpx+1] += self.mask
        state_img = state_img[self.rpx:-self.rpx, self.rpx:-self.rpx]
        state_img[np.nonzero(state_img)] = 1.0

        return state_img

    def get_action(self, state):
        action = None
        if self.random_episode is True:
            action = (torch.randn(6, dtype=torch.float)*5.0).abs().clip(0.0,10.0)
        else:
            action = (self.policy.select_action( state )
                      + np.random.normal(0, self.max_action * self.expl_noise, size = self.action_dim)
            ).clip(-self.max_action, self.max_action)
            action = np.abs( action )
        
        # Send action to hallucination module
        out_msg = VirtualCircles()
        c1 = VirtualCircle( radius = min(1.0, np.linalg.norm([action[0], action[1]])),
                            x      = action[0], 
                            y      = action[1], 
                            life   = rospy.Duration(action[2]))
        c2 = VirtualCircle( radius = min(1.0, np.linalg.norm([action[3], action[4]])), 
                            x      = action[3], 
                            y      = action[4], 
                            life   = rospy.Duration(action[5]))
        out_msg.circles = [c1, c2]
        self.pub_action.publish(out_msg)

        return action

    def get_reward(self, p1, p2):
        # Assume robot moves from p1 -> p2
        time = (p2.header.stamp - p1.header.stamp).to_sec()
        d_prev = self.dist(p1, self.goal.target_pose)
        d_curr = self.dist(p2, self.goal.target_pose)

        return -time + (d_prev - d_curr)

    def dist(self, point, goal):
        p1 = np.array([point.pose.position.x, point.pose.position.y])
        p2 = np.array([goal.pose.position.x, goal.pose.position.y])
        dist = np.linalg.norm(p1-p2)
        return dist

    def scan_cb(self, scan_msg):
        self.scan_msg = scan_msg

    def start_cb(self):
        self.finish = False
        self.start = rospy.Time.now()

        state = self.get_state( self.scan_msg )
        action = self.get_action( state )

        self.states.append(state)
        self.actions.append(action)

    def periodic_cb(self, feedback):
        if self.prev_location is not None:
            self.curr_location = feedback.base_position
            if (self.curr_location.header.stamp - self.prev_location.header.stamp).to_sec() >= 5.0:
                state = self.get_state( self.scan_msg )
                action = self.get_action( state )
                reward = self.get_reward(self.prev_location, self.curr_location)

                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.done.append(0)

                self.prev_location = copy.deepcopy(self.curr_location)
        else:
            self.prev_location = feedback.base_position
        pass

    def finish_cb(self, state, result):
        self.finish = True
        self.ttd = (rospy.Time.now() - self.start).to_sec()

        state = self.get_state( self.scan_msg )
        reward = self.get_reward( self.prev_location, self.curr_location) + 10.0 # Success advantage

        self.states.append(state)
        self.rewards.append(reward)
        self.done.append(1)
        rospy.loginfo("{} finished".format(self.entity_name))

        # pickle result
        result = {"state"   : self.states, 
                  "action"  : self.actions, 
                  "reward"  : self.rewards, 
                  "not_done": self.done}

        filename = "{}.pickle".format(str(uuid.uuid4()))
        with open( os.path.join(self.data_storage, filename), 'wb') as outfile:
            pickle.dump(result, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--explore", action="store_true", help="Run random episode")
    parser.add_argument("--max_action", default=10.0, type=float)
    parser.add_argument("--expl_noise", default=0.01, type=float)
    
    parser.add_argument("--x1",   default=-7.00, type=float)
    parser.add_argument("--y1",   default= 0.00, type=float)
    parser.add_argument("--yaw1", default= 0.00, type=float)
    
    parser.add_argument("--x2",   default= 7.00, type=float)
    parser.add_argument("--y2",   default= 0.00, type=float)
    parser.add_argument("--yaw2", default= 3.14, type=float)
    args = parser.parse_args()

    rospy.init_node('random_episode_py')
    rospy.sleep(1.0)

    marvin  = BWIbot('marvin',  random_episode=args.explore, max_action = args.max_action, expl_noise = args.expl_noise, action_dim = 6)
    roberto = BWIbot('roberto', random_episode=args.explore, max_action = args.max_action, expl_noise = args.expl_noise, action_dim = 6)

    print(f"Experiment number {sys.argv[1]}")
    rospy.sleep(2.0)

    marvin.move( args.x1, args.y1, args.yaw1)
    roberto.move(args.x2, args.y2, args.yaw2)

    begin = rospy.Time.now()
    while marvin.finish is not True or roberto.finish is not True:
        rospy.sleep(1.0)
        if (rospy.Time.now() - begin).to_sec() > 120:
            break
    # marvin.move_base.cancel_all_goals()
    # roberto.move_base.cancel_all_goals()