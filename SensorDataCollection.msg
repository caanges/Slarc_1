#SensorDataCollection.msg

#FUSED GPS DATA
uint64 global_position_time	# time since system start (microseconds)

float64 lat                 # Latitude, (degrees)
float64 lon                 # Longitude, (degrees)
float32 alt                 # Altitude AMSL, (meters)
float32 alt_ellipsoid       # Altitude above ellipsoid, (meters)

float32 terrain_alt         # Terrain altitude WGS84, (metres)
bool terrain_alt_valid		# Terrain altitude estimate is valid

# FUSED LOCAL POSITION

uint64 local_position_time  # time since system start (microseconds)

# Position in local NEU frame 
float32 x                   # North position in NEU earth-fixed frame, (metres)
float32 y                   # East position in NEU earth-fixed frame, (metres)
float32 z                   # Up position in NEU earth-fixed frame, (metres)

# Velocity in NEU frame
float32 vx 				# North velocity in NEU earth-fixed frame, (metres/sec)
float32 vy				# East velocity in NEU earth-fixed frame, (metres/sec)
float32 vz				# Down velocity in NEU earth-fixed frame, (metres/sec)

# Acceleration in NEU frame
float32 ax              # North velocity derivative in NEU earth-fixed frame, (metres/sec^2)
float32 ay              # East velocity derivative in NEU earth-fixed frame, (metres/sec^2)
float32 az              # Down velocity derivative in NEU earth-fixed frame, (metres/sec^2)

float32 heading         # Euler yaw angle transforming the tangent plane relative to NEU earth-fixed frame, -PI..+PI,  (radians)

# Position of reference point (local NEU frame origin) in global (GPS / WGS84) frame
float64 ref_lat         # Reference point latitude, (degrees)
float64 ref_lon         # Reference point longitude, (degrees)
float32 ref_alt         # Reference altitude AMSL, (metres)

# Distance to surface
float32 dist_bottom     # Distance from from bottom surface to ground, (metres)
bool dist_bottom_valid  # true if distance to bottom surface is valid

# ATTITUDE_QUATERNION

uint64 attitude_quaternion_time # time since system start (microseconds)

float32[4] q            # Quaternion rotation from the FRD body frame to the NED earth frame


