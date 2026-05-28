import cv2
import numpy as np
import math
import os





class PnP_processing_algot_DJI:

    

    def __init__(self):


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
        ] ,dtype = np.float32)


        self.camera_matrix = np.array([
            [2060.58337, 0, 1321.95959],
            [0, 2060.08079, 728.231313],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([
            [0.00669107, -0.1080931, -0.00540403, -0.00456828, 0.12385198]
        ], dtype=np.float32)

        self.img_w = 2720
        self.img_h = 1530


        self.conf_threshold = 0.8


    def CalculatePoseFromKeypoints(self, keypoints_2d, confs=None):
        valid_2d = []
        valid_3d = []

        for i, kp in enumerate(keypoints_2d):
            if i >= len(self.UGV_points_3D):
                break

            if confs is not None and confs[i] < self.conf_threshold:
                continue

            x_val, y_val = kp

            # If keypoints are normalized 0-1, convert to pixels
            if x_val <= 1.0 and y_val <= 1.0:
                x_val *= self.img_w
                y_val *= self.img_h

            valid_2d.append([x_val, y_val])
            valid_3d.append(self.UGV_points_3D[i])

        if len(valid_2d) < 4:
            print(f"Not enough points for PnP: {len(valid_2d)}")
            return None, None

        valid_2d = np.array(valid_2d, dtype=np.float32).reshape(-1, 1, 2)
        valid_3d = np.array(valid_3d, dtype=np.float32).reshape(-1, 1, 3)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            valid_3d,
            valid_2d,
            self.camera_matrix,
            self.dist_coeffs,
            iterationsCount=100,
            reprojectionError=20.0,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success:
            print("solvePnPRansac failed")
            return None, None

        return rvec, tvec

    def ExtractValuesFromYolo(self, file_path):
        # Reads the .txt file that yolo has stored all data in and converts it to a list for pnp
        Valid_2D_image_points = []
        matching_3D_object_points = []


        if not os.path.exists(file_path):
            return None, None
       
        with open(file_path, "r") as file:
            # Read everything and convert to a flat list of floats
            data = [float(val) for val in file.read().split()]

        # The first value is usually the class_id, followed by 4 bbox values (x, y, w, h)
        # If your file starts directly with the 4 bbox values, use start_index = 4
        # If it's standard YOLO (class_id + 4 bbox), use start_index = 5
        start_index = 5 
        keypoints_only = data[start_index:]

        # Process triplets: (x, y, confidence)
        for i in range(0, len(keypoints_only), 3):
            # Calculate which 3D point this corresponds to
            point_idx = i // 3

            if point_idx >= len(self.UGV_points_3D):
                break
            
            if i + 2 < len(keypoints_only):
                x_val = keypoints_only[i]
                y_val = keypoints_only[i+1]
                conf = keypoints_only[i+2]

                print(f"Point {point_idx}: X={x_val:.3f}, Y={y_val:.3f}, Conf={conf:.2f}")

                if conf >= self.conf_threshold:
                    x_pixel = x_val * self.img_w
                    y_pixel = y_val * self.img_h
                    Valid_2D_image_points.append([x_val, y_val])
                    matching_3D_object_points.append(self.UGV_points_3D[point_idx])
        print(f"Valid points meeting threshold: {len(Valid_2D_image_points)}")
        return np.array(matching_3D_object_points, dtype = np.float32), np.array(Valid_2D_image_points, dtype = np.float32)




    def CalculatePose(self, file_path):


        success = False #initialisering
        rvec, tvec, inliers = None, None, None

        Final_3D_point, Final_2D_point = self.ExtractValuesFromYolo(file_path)# File pathen är faktiska pathen till där yolo .txt filen befinner sig
       
        if Final_3D_point is None or len(Final_2D_point) < 4:
            # Added a print statement to show the exact number of detected points
            print(f"Not enough points detected in {os.path.basename(file_path)} "
                  f"(detected: {0 if Final_2D_point is None else len(Final_2D_point)}, required: 4)")
            return None, None

        # Convert points to the correct format and data type
        Final_3D_point = np.array(Final_3D_point, dtype=np.float32).reshape(-1,1,3)
        Final_2D_point = np.array(Final_2D_point, dtype = np.float32).reshape(-1,1,2)

        try:    
            success, rvec, tvec, inliers = cv2.solvePnPRansac(Final_3D_point, Final_2D_point, self.camera_matrix, 
            self.dist_coeffs, iterationsCount = 100, reprojectionError = 20.0, flags = cv2.SOLVEPNP_EPNP ) # Här utförs självaste uträkningen av rvec och tvec
                                                                                                            # reprojectionError var 8.0 innan
                                                                                                            # flags var SOLVEPNP_ITERATIVE innan                                                
        # Ensure the solve was successful and the sizes are exactly 3
            if success and rvec is not None and tvec is not None:
                rvec_arr = np.array(rvec, dtype = np.float32).flatten()
                tvec_arr = np.array(tvec, dtype = np.float32).flatten()

                if rvec_arr.size == 3 and tvec_arr.size == 3:
                    return rvec, tvec
                else:
                    print(f"Pose estimation failed for {os.path.basename(file_path)}: "
                          f"Vector sizes invalid (rvec size: {rvec_arr.size}, tvec size: {tvec_arr.size}).")
            else:
                # This print will catch the case where solvePnPRansac returns success=False
                print(f"Pose estimation failed for {os.path.basename(file_path)}: "
                      f"solvePnPRansac returned success={success}, rvec={rvec}, tvec={tvec}.")
                return None, None
            
            
        
        except Exception as e:
            print(f"Error in solvePnPRansac: {e}")
            return None, None


        '''
        if len(Final_2D_point) >= 4 and Final_3D_point is not None: # Det var or här innan #Ska det inte finnas final_3d_point is not None?
            success, rvec, tvec, inliers = cv2.solvePnPRansac(Final_3D_point, Final_2D_point, self.camera_matrix, 
            self.dist_coeffs, iterationsCount = 100, reprojectionError = 8.0, flags = cv2.SOLVEPNP_ITERATIVE ) # Här utförs självaste uträkningen av rvec och tvec
        else:
            print("Not enough points detected to solve for pose.")
        '''

        #return (rvec, tvec)if success and inliers is not None else (None, None)
   




def main(): 
    folder_path = r"C:\Users\een23013\Desktop\PNPtest"
    image_folder = r"C:\Users\een23013\Desktop\PNPtest"

    # Instantiate the processor
    processor = PnP_processing()

    if not os.path.exists(folder_path): # Till exempel "./yolo_results"
        print(f"Folder '{folder_path}' does not exist.")
        return

    axis = np.float32([
        [0,0,0],
        [15,0,0],
        [0,15,0],
        [0,0,15]
    ])

    for filename in os.listdir(folder_path): # filename är temporärt namn av yolo .txt filen, namnet är inte så viktigt, det viktiga är att det är en .txt fil
        if filename.endswith(".txt"): # Anledningen till varför man inte döper till riktiga filnamnet är för att man måste då byta filnman hela tiden efter man kör yolo modellen på nytt
            full_path = os.path.join(folder_path, filename)
            base_name = os.path.splitext(filename)[0]

            img_path = os.path.join(image_folder, f"{base_name}.png")

            rvec, tvec = processor.CalculatePose(full_path)

            if rvec is not None and tvec is not None:
                # FIX: Force arrays into the strict shape/type cv2.projectPoints requires
                # Safely flatten and check the size of rvec and tvec
                rvec_arr = np.array(rvec, dtype=np.float32).flatten()
                tvec_arr = np.array(tvec, dtype=np.float32).flatten()

                # FIX: Skip if vectors don't have exactly 3 elements
                if rvec_arr.size !=3 or tvec_arr.size !=3:
                    print(f"Skipping frame: rvec/tvec size invalid (rvec={rvec_arr.size}, tvec={tvec_arr.size}).")
                    continue
                
                rvec_safe = rvec_arr.reshape(3, 1)
                tvec_safe = tvec_arr.reshape(3, 1)

                try: # Project the 3D points/axes into 2D image coordinates
                    imgpts, jacobian = cv2.projectPoints(axis, rvec_safe, tvec_safe, processor.camera_matrix, processor.dist_coeffs) # cv2.projectPoints för att verifiera rvec och tvec

                    # Print the raw arrays
                    print("Rotation Vector (rvec):\n", rvec) # Angle in radians
                    print("Translation Vector (tvec):\n", tvec)
                    
                    # Print cleaner, one-line versions
                    print(f"Position (X, Y, Z): {tvec.flatten()}")

                    if os.path.exists(img_path):
                        #img = cv2.imread(img_path)        
                        img_array = np.fromfile(img_path, np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    else:
                        print(f"No corresponding image found for {filename}. Using blank background")
                        img = np.zeros((800,1280,3), dtype=np.uint8)

                    #Här ska plotting ske
                    imgpts = imgpts.astype(int)

                    o, x, y, z = imgpts.reshape(-1,2) # x,y,z här är annorlunda från tvec x,y,z

                    cv2.line(img, tuple(o), tuple(x), (0,0,255), 3) # X - Red
                    cv2.line(img, tuple(o), tuple(y), (0,255,0), 3) # Y - Green
                    cv2.line(img, tuple(o), tuple(z), (255,0,0), 3) # Z - Blue
                    
                    cv2.imshow("Pose estimation", img) #displaya bilden, Innan stod det cv2_imshow med udnerstreck vilket kommer från google collab

                except Exception as e:
                    print(f"Error projecting points: {e}")
            else:
                print(f"Pose estimation skipped for: {filename}")

            cv2.waitKey(0) #Kolla om detta behövs

    cv2.destroyAllWindows()

    '''
    # Display the plotted image
    try:
        from google.colab.patches import cv2_imshow
        cv2_imshow(img)
    except ImportError:
        cv2.imshow("Pose estimation", img)
        cv2.waitKey(0)
    '''


if __name__ == "__main__":
    main()




  