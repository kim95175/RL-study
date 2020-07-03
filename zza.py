#! /usr/bin/env python

import rospy
import utm
import math
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from FusionEKF import FusionEKF
from EKF import EKF
from tf.transformations import euler_from_quaternion

def gps_callback(msg):
    global result_before
    UTM_POSI = fusionEKF.process_GPS(msg)
    UTM_POS = UTM_POSI[0], UTM_POSI[1]
    print(UTM_POS)
    print("gg")
    state_udt = ekf.update(result_before, 1/50.0, np.matrix(UTM_POS))
    result_before = state_udt
    print(state_udt)

    #return UTM_POS

def odom_callback(msg):
    pass

def imu_callback(msg):
    Yaw, Yawrate = fusionEKF.process_IMU(msg)


    #return Yawrate

def hdg_callback(msg):
    pass


rospy.init_node('ros_sub')

# create FusionEKF object here
fusionEKF = FusionEKF()
ekf = EKF()
result_before = []




# create subscriber for differnt sensor topics
odom_sub = rospy.Subscriber('/mavros/global_position/local', Odometry, odom_callback)
#imu_sub = rospy.Subscriber('/imu/data_raw', Odometry, imu_callback)
imu_sub = rospy.Subscriber('/imu/data_raw', Imu, imu_callback)
gps_sub = rospy.Subscriber('/fix', NavSatFix, gps_callback)
hdg_sub = rospy.Subscriber('/mavros/global_position/compass_hdg', Float64, hdg_callback)


   
#x = [UTM_POS[0], UTM_POS[1], 0.1, 1, Yawrate]
#state_udt = ekf.update(state_prd, 1/50.0, UTM_POS)
    

#state_prd = ekf.predict(Yawrate, 1/50.0)
#state_udt = ekf.update(state_prd, 1/50.0, UTM_POS)


#print(state_udt)

rospy.spin()
