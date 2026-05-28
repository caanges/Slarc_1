import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
import depthai as dai
import time
import math
from datetime import timedelta












class UGVOnDeviceNode(Node):
 def __init__(self):
     super().__init__('ugv_on_device_vision_node')








     # Bilddimensioner (640x640 kvadratiskt för att matcha YOLOv8-pose-blob)
     self.img_w = 640
     self.img_h = 640 








     #Datat skickas ut på våran topic /ugv_pose för andra möjliga applikationer
     self.pose_publisher = self.create_publisher(PoseStamped, '/ugv_pose', 1)
     self.get_logger().info("UGV On-Device Vision Node har startat.")








     # Tröskelvärden för AI:n och Non-Maximum Suppression
     self.conf_threshold = 0.4
     self.nms_iou_threshold = 0.45
     self.num_keypoints = 13








     # Dina 13 stycken 3D-punkter på UGV:n
     self.UGV_points_3D = np.array([
         [-5.25, -40.4, 4.5],
           [9.75, -29.0, -2.5],
           [-9.75, -29.0, -2.5],
           [-22.5, -16.25, 0.0],
           [22.5, -16.25, 0.0],
           [-22.5, 16.25, 0.0],
           [-15.5, 11.75, 6.0],
           [15.5, 11.75, 6.0],
           [22.5, 16.25, 0.0],
           [-9.75, 29.25, -2.5],
           [9.75, 29.25, -2.5],
           [-8.5, 43.25, -5.0],
           [8.5, 43.25, -5.0]
     ], dtype=np.float32)








     # Lager-namnen som matchar din exporterade blob
     self.layer_names = [
         "output1_yolov8", "kpt_output1",
         "output2_yolov8", "kpt_output2",
         "output3_yolov8", "kpt_output3"
     ]








     # B. DEPTHAI PIPELINE SETUP
     self.pipeline = self.create_pipeline()
  
     # C. STARTA ENHETEN
     self.device = dai.Device(self.pipeline)








     # D. FELSÄKER FABRIKSKALIBRERING (CAM_B-Fix)
     self.get_logger().info("Hämtar fabrikskalibrering från OAK-D...")
     calibData = self.device.readCalibration()








     try:
         self.camera_matrix = np.array(
             calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, resizeWidth=self.img_w, resizeHeight=self.img_h),
             dtype=np.float32
         )
         self.dist_coeffs = np.array(
             calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B),
             dtype=np.float32
         )
         self.get_logger().info("Lyckades hämta kalibrering för CAM_B!")
     except Exception as e:
         self.get_logger().error(f"Kunde inte hämta kalibrering för CAM_B: {e}")
         raise e








     # E. SKAPA UTGÅNGSKÖ (maxSize=1, blocking=False tvingar fram lägsta möjliga latens)
     self.q_sync = self.device.getOutputQueue(name="synced_data", maxSize=1, blocking=False)








     # F. STARTA TIMER (Körs så snabbt som möjligt på huvudtråden)
     self.timer = self.create_timer(0.001, self.run_loop)
     self.get_logger().info("Ultra-low latency hardware synced loop startad.")
  
 def create_pipeline(self):
     pipeline = dai.Pipeline()








     # 1. Kamera-inställningar
     cam = pipeline.createColorCamera()
     cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
     cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
     cam.setPreviewSize(self.img_w, self.img_h)
     cam.setPreviewKeepAspectRatio(True)
     cam.setInterleaved(False)
     cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
     cam.setFps(30)








     # 2. Neural Network-nod
     nn = pipeline.createNeuralNetwork()
     #Lägg in egen directory för där .blob filen (weights) befinner sig
     nn.setBlobPath('/home/slarc/Downloads/BlobWeights/bestmixed.blob')
     cam.preview.link(nn.input)








     # 3. HÅRDVARUSYNKRONISERING
     sync = pipeline.create(dai.node.Sync)
  
     # Vi länkar in strömmarna till valfria unika ingångsnamn
     nn.passthrough.link(sync.inputs["rgb_stream"])
     nn.out.link(sync.inputs["nn_stream"])








     # 4. Skapa utgångsport till din NUC
     xout_grouped = pipeline.createXLinkOut()
     xout_grouped.setStreamName("synced_data")
     sync.out.link(xout_grouped.input)








     return pipeline








 def run_loop(self):
     # Töm kön helt så vi bara bearbetar det absolut senaste synkroniserade paketet
     latest_packet = None
     while True:
         check_packet = self.q_sync.tryGet()
         if check_packet is None:
             break
         latest_packet = check_packet








     if latest_packet is not None:
         # Hämta meddelandegruppen (MessageGroup) från Sync-noden
         # Vi plockar ut bild och AI-data med de namngivna portarna vi definierade i pipelinen
         rgb_msg = latest_packet["rgb_stream"]
         nn_msg = latest_packet["nn_stream"]








         # OAK-D HÅRDVARULATENS (Inkluderar YOLO-inferens)
         capture_time = rgb_msg.getTimestamp()
         current_time = timedelta(seconds=time.monotonic())
         oak_d_latency_ms = (current_time - capture_time).total_seconds() * 1000.0








         # KOD- & PNP-LATENS PÅ NUC
         start_time = time.perf_counter()




         # Konvertera till OpenCV-format (Garanterat i perfekt synk med AI-datan)
         frame = rgb_msg.getCvFrame()








         # Plocka ut lagren från AI:n
         outputs = {}
         for name in self.layer_names:
             outputs[name] = np.array(nn_msg.getLayerFp16(name), dtype=np.float32)








         # Avkoda koordinater och rita bounding boxes/keypoints
         final_3D, final_2D = self.decode_multi_output_yolov8_pose(outputs, frame)
      
         # Kontrollera att vi fick tillräckligt med giltiga punkter för PnP
         if final_3D is not None and len(final_2D) >= 4:
             rvec, tvec = self.calculate_pnp(final_3D, final_2D)








             if tvec is not None and rvec is not None:
                 self.publish_pose(tvec, rvec)
                 self.draw_axes(frame, rvec, tvec) # Utritning av axlar








         # Stoppa NUC-timern
         end_time = time.perf_counter()
         code_latency_ms = (end_time - start_time) * 1000.0
         # Logga resultatet var 30:e frame (ca 1 gång per sekund)




         #Ifall man vill printa ut latens tiden av alla faser av systemet, avkommentera bort då följande print
         #if rgb_msg.getSequenceNum() % 30 == 0:
             #self.get_logger().info(f"OAK-D Total (Kamera+YOLO+USB): {oak_d_latency_ms:.1f}ms | Kod+PnP på NUC: {code_latency_ms:.1f}ms")




         # Visa live-bildfönstret
         cv2.imshow("OAK-D Hardware Synced Tracker", frame)
         cv2.waitKey(1)








 def decode_multi_output_yolov8_pose(self, outputs, frame=None):
     all_boxes = []
     all_scores = []
     all_kpts_list = []
  
     pairs = [
         ("output1_yolov8", "kpt_output1", 8),
         ("output2_yolov8", "kpt_output2", 16),
         ("output3_yolov8", "kpt_output3", 32),
     ]
  
     logit_thresh = math.log(self.conf_threshold / (1.0 - self.conf_threshold))
  
     for box_name, kpt_name, stride in pairs:
         if box_name not in outputs or kpt_name not in outputs:
             continue
          
         box_raw = outputs[box_name]
         kpt_raw = outputs[kpt_name]
      
         if box_raw.size == 0 or kpt_raw.size == 0:
             continue








         num_cells = box_raw.size // 6
         boxes = box_raw.reshape(6, num_cells)
         keypoints_raw = kpt_raw.reshape(39, num_cells)








         max_logits = np.maximum(boxes[4, :], boxes[5, :])
         valid_mask = max_logits >= logit_thresh
         valid_indices = np.where(valid_mask)[0]
      
         if len(valid_indices) == 0:
             continue
          
         valid_boxes = boxes[:, valid_indices]
         valid_kpts = keypoints_raw[:, valid_indices]
         valid_logits = max_logits[valid_indices]
      
         valid_confs = 1.0 / (1.0 + np.exp(-np.clip(valid_logits, -80, 20)))
      
         cx = valid_boxes[0, :] * stride
         cy = valid_boxes[1, :] * stride
         w = valid_boxes[2, :] * stride
         h = valid_boxes[3, :] * stride
      
         x1 = cx - w / 2
         y1 = cy - h / 2
      
         kx = valid_kpts[0::3, :]
         ky = valid_kpts[1::3, :]
         k_logits = valid_kpts[2::3, :]
         k_confs = 1.0 / (1.0 + np.exp(-np.clip(k_logits, -80, 20)))
      
         for j in range(len(valid_indices)):
             all_boxes.append([int(x1[j]), int(y1[j]), int(w[j]), int(h[j])])
             all_scores.append(float(valid_confs[j]))
          
             det_kpts = []
             for k in range(self.num_keypoints):
                 kc = float(k_confs[k, j])
                 if kc >= 0.5:
                     det_kpts.append((float(kx[k, j]), float(ky[k, j]), kc))
                 else:
                     det_kpts.append((0.0, 0.0, 0.0))
             all_kpts_list.append(det_kpts)








     if len(all_boxes) == 0:
         return None, None








     indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, self.conf_threshold, self.nms_iou_threshold)
  
     if len(indices) == 0:
         return None, None
      
     best_idx = indices[0]
     if isinstance(best_idx, (list, np.ndarray)):
         best_idx = best_idx[0]
      
     best_box = all_boxes[best_idx]
     best_score = all_scores[best_idx]
     best_kpts = all_kpts_list[best_idx]
  
     valid_2D_points = []
     matching_3D_points = []
  
     if frame is not None:
         bx1, by1, bw, bh = best_box
         bx2, by2 = bx1 + bw, by1 + bh
         cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 255), 2)
         cv2.putText(frame, f"UGV: {best_score:.2f}", (bx1, max(by1 - 10, 20)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
  
     for k, (px, py, pc) in enumerate(best_kpts):
         if pc >= 0.5 and 0 <= px < self.img_w and 0 <= py < self.img_h:
             valid_2D_points.append([px, py])
             matching_3D_points.append(self.UGV_points_3D[k])
          
             if frame is not None:
                 cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)
              
     return np.array(matching_3D_points, dtype=np.float32), np.array(valid_2D_points, dtype=np.float32)








 def calculate_pnp(self, object_points, image_points):
     object_points = object_points.reshape(-1, 1, 3)
     image_points = image_points.reshape(-1, 1, 2)
     try: 
         success, rvec, tvec, inliers = cv2.solvePnPRansac(
             object_points, image_points, self.camera_matrix,
             self.dist_coeffs, iterationsCount=100, reprojectionError=20.0,
             flags=cv2.SOLVEPNP_EPNP
         )
         if success and rvec is not None and tvec is not None:
             return rvec, tvec
     except Exception as e:
         self.get_logger().error(f"Error i solvePnPRansac: {e}")
     return None, None








 def publish_pose(self, tvec, rvec):
     msg = PoseStamped()
     msg.header.stamp = self.get_clock().now().to_msg()
     msg.header.frame_id = "camera_link"
     msg.pose.position.x = float(tvec[0][0])
     msg.pose.position.y = float(tvec[1][0])
     msg.pose.position.z = float(tvec[2][0])
     # Rotation: Konvertering av rvec (Rodrigues) till rotationsmatris, så att orienteringen i ROS2 är rätt. Det konverteras så att ROS2 förstår
     rotation_matrix, _ = cv2.Rodrigues(rvec) 
     #Konvertering av rotationsmatris till kvaternion så att det passar ROS2
     #Vi bygger 4x4 matris för transformationen
     quat = self.rotation_matrix_to_quaternion(rotation_matrix)
     msg.pose.orientation.x = quat[0]
     msg.pose.orientation.y = quat[1]
     msg.pose.orientation.z = quat[2]
     msg.pose.orientation.w = quat[3]
    


     self.pose_publisher.publish(msg)




 def rotation_matrix_to_quaternion(self, matrix):
     # En matematisk omvandling från matris till kvaternion
     trace = matrix[0,0] + matrix[1,1] + matrix[2,2]
     if trace > 0:
         root = np.sqrt(trace + 1.0) * 2
         quaternion_w = 0.25 * root
         quaternion_x = (matrix[2,1] - matrix[1,2]) / root
         quaternion_y = (matrix[0,2] - matrix[2,0]) / root
         quaternion_z = (matrix[1,0] - matrix[0,1]) / root
     elif (matrix[0,0] > matrix[1,1]) and (matrix[0,0] > matrix[2,2]):
         root = np.sqrt(1.0 + matrix[0,0] - matrix[1,1] - matrix[2,2]) * 2
         quaternion_w = (matrix[2,1] - matrix[1,2]) / root
         quaternion_x = 0.25 * root
         quaternion_y = (matrix[0,1] + matrix[1,0]) / root
         quaternion_z = (matrix[0,2] + matrix[2,0]) / root
     elif matrix[1,1] > matrix[2,2]:
         root = np.sqrt(1.0 + matrix[1,1] - matrix[0,0] - matrix[2,2]) * 2
         quaternion_w = (matrix[0,2] - matrix[2,0]) / root
         quaternion_x = (matrix[0,1] + matrix[1,0]) / root
         quaternion_y = 0.25 * root
         quaternion_z = (matrix[1,2] + matrix[2,1]) / root
     else:
         root = np.sqrt(1.0 + matrix[2,2] - matrix[0,0] - matrix[1,1]) * 2
         quaternion_w = (matrix[1,0] - matrix[0,1]) / root
         quaternion_x = (matrix[0,2] + matrix[2,0]) / root
         quaternion_y = (matrix[1,2] + matrix[2,1]) / root
         quaternion_z = 0.25 * root
        
     return [quaternion_x, quaternion_y, quaternion_z, quaternion_w]
 
 #Kod-funktion till för att plotta axlarna i live-video feeden. Avkommentera bort ifall det inte önskas
 def draw_axes(self, frame, rvec, tvec):
     axis = np.float32([[0,0,0], [15,0,0], [0,15,0], [0,0,15]])
     imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
     imgpts = imgpts.astype(int).reshape(-1, 2)
     o, x, y, z = imgpts
     cv2.line(frame, tuple(o), tuple(x), (0,0,255), 3) # X - Röd
     cv2.line(frame, tuple(o), tuple(y), (0,255,0), 3) # Y - Grön
     cv2.line(frame, tuple(o), tuple(z), (255,0,0), 3) # Z - Blå
















def main(args=None):
 rclpy.init(args=args)
 node = UGVOnDeviceNode()
 try:
     rclpy.spin(node)
 except KeyboardInterrupt:
     pass
 finally:
     cv2.destroyAllWindows()
     node.destroy_node()
     rclpy.shutdown()








if __name__ == '__main__':
 main()
