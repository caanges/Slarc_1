import sys
import os




#  Kapa tillbaka depthai från ROS
if 'VIRTUAL_ENV' in os.environ: # Här frågar vi operativsystemet: "Körs jag i en virtuell miljö?". Om svaret är ja, får vi reda på exakt var den ligger.
  #Här pekar vi ut den exakta mappen där din fungerande depthai ligger (t.ex. /home/slarc/depthai_venv/lib/python3.12/site-packages).
  venv_path = os.path.join(os.environ['VIRTUAL_ENV'], 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
  if venv_path in sys.path: #Om mappen redan finns i listan, tas den bort först. Detta förhindrar att vi får dubbletter eller hamnar i konflikt med oss själva
      sys.path.remove(venv_path)
  sys.path.insert(0, venv_path)  # Tvinga Python att leta här FÖRST. sys.path är en lista där Python letar efter moduler, och det letas alltid från index 0 och framåt.
   # Insert(0, ...) puttar in den fungerande depthai längst fram i kön                         


import depthai as dai # Python tittar på listan, se versionen vid index 0, och ladda den direkt, innan den hinner ens titta på de trasiga versionerna som ROS2 försökte packa på oss
print(f"\n[DIAGNOSTIK] Laddade depthai från: {dai.__file__}\n")




import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import timedelta




class UGVVisionNode(Node):




  def __init__(self):
      super().__init__('ugv_vision_node')




      # 3D-punkter för UGV
      self.UGV_points_3D = np.array([
          [5.25, 40.4, 4.5], [-9.75, 29.0, -2.5], [9.75, 29.0, -2.5],
          [22.5, 16.25, 0.0], [-22.5, 16.25, 0.0], [22.5, -16.25, 0.0],
          [15.5, -11.75, 6.0], [-15.5, -11.75, 6.0], [-22.5, -16.25, 0.0],
          [9.75, -29.25, -2.5], [-9.75, -29.25, -2.5], [8.5, -43.25, -5.0],
          [-8.5, -43.25, -5.0]
      ], dtype=np.float32)




      self.conf_threshold = 0.8




      # Ladda YOLO (OpenVINO)
      model_path = '/home/slarc/Downloads/bridge_weights/best_real_world_openvino_model'
      self.model = YOLO(model_path, task='pose')    




      self.get_logger().info("Startar anslutning till OAK-D-SR...")
    
      pipeline = dai.Pipeline()




      cam = pipeline.create(dai.node.ColorCamera)
      cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
      cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
      cam.setInterleaved(False)
      cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
      cam.setFps(30)




      xout_rgb = pipeline.create(dai.node.XLinkOut)
      xout_rgb.setStreamName("rgb")
    
      cam.video.link(xout_rgb.input)




      # Starta enheten
      self.device = dai.Device(pipeline)
      self.queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    
      # Hämta kalibrering för 1280x800
      calibData = self.device.readCalibration()
      intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 1280, 800)
      self.camera_matrix = np.array(intrinsics, dtype=np.float32) # Beskriver kamerans inre egenskaper, förklarar hur 3D rymden "skalas ner" till vår bildruta
      self.dist_coeffs = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B), dtype=np.float32) #Beskriver linsens avvikelser
    
      #print("\n" + "="*40)
      #print("Kamera-matris (Camera Matrix):")
      #print(repr(self.camera_matrix))
      #print("\nDistorsionskoefficienter (Dist Coeffs):")
      #print(repr(self.dist_coeffs))
      #print("="*40 + "\n")


      self.get_logger().info("Kamera ansluten och kalibrering hämtad!")




      self.pose_publisher = self.create_publisher(PoseStamped, '/ugv_pose', 10)
      self.timer = self.create_timer(1.0 / 30.0, self.process_frame)
      self.get_logger().info("UGV Vision Node rullar! Letar efter bilder...")




  def ExtractValuesFromResults(self, results):
      Valid_2D_image_points = []
      matching_3D_object_points = []




      # Kolla om listan är tom eller om inga skelett/keypoints hittades överhuvudtaget
      if (len(results) == 0 or
          results[0].keypoints is None or
          results[0].keypoints.data is None or
          results[0].keypoints.data.shape[0] == 0):
          return None, None




      # Nu är det säkert att läsa av index 0!
      kp_data = results[0].keypoints.data[0].cpu().numpy()
      for i in range(len(kp_data)):
          if i >= len(self.UGV_points_3D):
              break
          x, y, conf = kp_data[i]




          if conf >= self.conf_threshold:
              Valid_2D_image_points.append([x, y])
              matching_3D_object_points.append(self.UGV_points_3D[i])




      return np.array(matching_3D_object_points, dtype=np.float32), np.array(Valid_2D_image_points, dtype=np.float32)




  def process_frame(self):
      in_rgb = self.queue.tryGet()
    
      if in_rgb is not None:
          frame = in_rgb.getCvFrame()


           # Kameran stämplade bilden exakt när den togs
          capture_time = in_rgb.getTimestamp()
          # Vad är systemklockan just nu när vi tog emot bilden? (På Linux använder DepthAI CLOCK_MONOTONIC)
          current_time = timedelta(seconds=time.monotonic())


          # Räkna ut skillnaden i millisekunder
          oak_d_latency = (current_time - capture_time).total_seconds() * 1000.0




          #Starta kod-timern
          start_time = time.perf_counter()


          # Kör YOLO (OpenVINO)
          results = self.model.predict(frame, verbose=False, device ='intel:gpu')


          speed = results[0].speed
          #inference_time = speed['inference']
          yolo_total = speed['preprocess'] + speed['inference'] + speed['postprocess']


         


          # Beräkna Pose
          Final_3D_point, Final_2D_point = self.ExtractValuesFromResults(results)
         
         
      
          


          if Final_3D_point is not None and len(Final_2D_point) >= 4:
              Final_3D_point = Final_3D_point.reshape(-1, 1, 3)
              Final_2D_point = Final_2D_point.reshape(-1, 1, 2)




              try:  
                  success, rvec, tvec, inliers = cv2.solvePnPRansac(
                      Final_3D_point, Final_2D_point, self.camera_matrix,
                      self.dist_coeffs, iterationsCount=100, reprojectionError=20.0,
                      flags=cv2.SOLVEPNP_EPNP
                  )
                  if success and rvec is not None and tvec is not None:
                      self.publish_pose(tvec)
                      self.draw_axes(frame, rvec, tvec)
              except Exception as e:
                  self.get_logger().error(f"PnP Error: {e}")


          # Stoppa kod timern
          end_time = time.perf_counter()
          code_latency_ms = (end_time - start_time) * 1000


          if self.get_clock().now().nanoseconds % 30 == 0:
               self.get_logger().info(f"OAK-D+USB (receiving image): {oak_d_latency:.1f}ms | YOLO: {yolo_total:.1f}ms | kod+PnP: {code_latency_ms:.1f}")


          # Visa live-feeden
          cv2.imshow("ROS 2 UGV Tracker", frame)
          cv2.waitKey(1)




  def publish_pose(self, tvec):
      msg = PoseStamped()
      msg.header.stamp = self.get_clock().now().to_msg()
      msg.header.frame_id = "camera_link"




      msg.pose.position.x = float(tvec[0][0])
      msg.pose.position.y = float(tvec[1][0])
      msg.pose.position.z = float(tvec[2][0])
   
      self.pose_publisher.publish(msg)




  def draw_axes(self, frame, rvec, tvec):
      axis = np.float32([[0,0,0], [-15,0,0], [0,-15,0], [0,0,15]])
      imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
      imgpts = imgpts.astype(int).reshape(-1, 2)
      o, x, y, z = imgpts
      cv2.line(frame, tuple(o), tuple(x), (0,0,255), 3)
      cv2.line(frame, tuple(o), tuple(y), (0,255,0), 3)
      cv2.line(frame, tuple(o), tuple(z), (255,0,0), 3)




def main(args=None):
  rclpy.init(args=args)
  node = UGVVisionNode()
  try:
      rclpy.spin(node)
  except KeyboardInterrupt:
      pass
  finally:
      cv2.destroyAllWindows()
      if hasattr(node, 'device'):
          node.device.close()
      node.destroy_node()
      rclpy.shutdown()




if __name__ == '__main__':
  main()


