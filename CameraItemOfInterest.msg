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

uint16 camera_publisher_id
bool cam_object_in_frame
bool cam_centered_in_frame
uint16 object_type                          #typ of object that was discovered
uint64 time                                 #time of discovery
float32 object_heading                      #heading of the discovered object
geometry_msgs/Point object_location         # point(local or global) where the object was discovered, 
geometry_msgs/Vector3 object_velocity       # Velocity of the object

