import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point, Vector3
from rclpy.qos import QoSProfile
import time
import cv2
import numpy as np 
from custom_drone_interfaces.msg import SensorDataCollection, CameraItemOfInterest, StereoCameraControl

class HeatObjects(Node):
	def __init__(self):
		super().__init__('heat_objects')
		self.cap = cv2.VideoCapture(0) # Device number system specific!
		self.downward_stereo_publish_data = False
		self.SensorData = SensorDataCollection()

		#Subscribers
		self.subscription_control = self.create_subscription(StereoCameraControl, '/mc/downward_stereo_camera_settings', self.downward_stereo_control_callback, 10)
		self.sensor_data_subscriber = self.create_subscription(SensorDataCollection, '/mc/sensor_data', self.listener_vehicle_sensors, QoSProfile(depth=10))

		#Publishers
		self.publisher_itemOfInterest = self.create_publisher(CameraItemOfInterest, '/mc/thermal_item_of_interest', 10)

		#Timers
		self.timer_period = 0.5
		self.timer = self.create_timer(self.timer_period, self.timer_callback)

	def downward_stereo_control_callback(self, stereoCameraSettings):
		#-----------------------------------------------------
		#Fyll ut detta. Vilka inställningar behöver ni?
		#Vilka ytterliggare fält mer än nedanstående behövs?
		#DepthCameraControl.msg
		#bool active
		#-----------------------------------------------------

		self.downward_stereo_publish_data = stereoCameraSettings.active

	def listener_vehicle_sensors(self, msg):
		try:
			self.SensorData = msg
		except Exception as e:
			self.get_logger().info(f'Error in listener_vehicle_sensors: {e}')

	def timer_callback(self):
		r, frame = self.cap.read()
		if not r:
			self.get_logger().info(f'Error in reading camera!')
			return
		
		# Publish/store image, or other functionality, whatever you find best. 
		if self.downward_stereo_publish_data:
			self.publish_image(frame)

	def publish_image(self, frame):
		#------------------------------------
		#Detta är CameraItemOfInterest:
		#CameraItemOfInterest.msg
		#object_type
		# 0 - Unspecified
		# 1 - Human
		# 2 - Small animal
		# 3 - Medium animal
		# 4 - Large animal
		# 5 - Bike/moped/motorcycle
		# 6 - Quad/snowmobile
		# 7 - Car
		# 8 - Truck/Bus
		# 9 - Unspecified Vehicle
		# 10 - Friendly UGV

		#uint16 camera_publisher_id				
		#bool cam_object_in_frame
		#bool cam_centered_in_frame
		#uint16 object_type                          #typ of object that was discovered
		#uint64 time                                 #time of discovery
		#float32 object_heading                      #heading of the discovered object
		#geometry_msgs/Point object_location         # point(local or global) where the object was discovered, 
		#geometry_msgs/Vector3 object_velocity       # Velocity of the object
		#-------------------------------------
		dt = int(time.time())
		p_msg = CameraItemOfInterest()
		
		p_msg.object_type = 10

		location = Point()
		location.x = self.SensorData.lat
		location.y = self.SensorData.lon
		location.z = self.SensorData.alt

		velocity = Vector3() #You have to decide if you want this or not. 

		p_msg.time = dt
		p_msg.location = location
		p_msg.velocity = velocity

		position_data = str(location.x) + "-" + str(location.y) + "-" + str(location.z)

		sensor_data = "{}_{}".format(dt,position_data)

		RAW_name = "img/stereo{}_{}.png".format(sensor_data,"-RAW")


		cv2.imwrite(RAW_name, frame)
		
		self.publisher_itemOfInterest.publish(p_msg)
		#cv2.imshow('image') #Only for debug
		cv2.waitKey(1)

def main(args=None):

	if not rclpy.ok():
		rclpy.init(args=args)
	thermal_camera_nide = HeatObjects()
	try:
		rclpy.spin(thermal_camera_nide)
	except KeyboardInterrupt:
		pass
	finally:
		thermal_camera_nide.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()

if __name__ == '__main__':
	main()