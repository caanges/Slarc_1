import cv2
import numpy as np
import math
import os





class PnP_processing:


    def __init__(self):


        UGV_points_3D = np.array([
            [1,0,0],
            [2,0,0],
            [3,0,0],
            [4,0,0],
            [5,0,0],
            [6,0,0],
            [7,0,0],
            [8,0,0],
            [9,0,0],
            [10,0,0],
            [11,0,0],
            [12,0,0],
            [13,0,0]
        ] ,dtype = np.float32)


        focal_length = 1280 # Dessa värden måste ändras utifrån våran kamera
        center = (1280 / 2, 800 / 2) # Denna med
        self.camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
        ], dtype=np.float32)


        self.dist_coeffs = np.zeros((4, 1))


        self.conf_threshold = 0.8


        self.img_w = 1280 # OBS! Exempel värden
        self.img_h = 800


    def ExtractValuesFromYolo(self, file_path):
        # Reads the .txt file that yolo has stored all data in and converts it to a list for pnp
        Valid_2D_image_points = []
        matching_3D_object_points = []


        if not os.path.exists(file_path):
            return None, None
       
        with open("filnamnet där yolo datat är sparat", "r") as file:


            for line in file:
                # Assumed that YOLO format: class_id x_center y_center width height confidence
                parts = line.split()
                if len(parts) < 6: continue # Skip broken lines


                class_id = int(parts[0]) # I denna for loop så läses det in en linje för sig i .txt filen och därav så tar man fram dess class_id, conf, x, y och om dess confidence är tillräcklig så kan den skickas in i accepterade 2d punkter, och matchande 3d punkten registreras då också med samma class_id
                # Convert normalized (0-1) to pixels
                x_pixel = float(parts[1]) * self.img_w
                y_pixel = float(parts[2]) * self.img_h
                conf = float(parts[5])


                if conf >= self.conf_threshold:
                    Valid_2D_image_points.append([x_pixel, y_pixel])
                    matching_3D_object_points.append(self.UGV_points_3D[class_id])


        return np.array(matching_3D_object_points, dtype = np.float32), np.array(Valid_2D_image_points, dtype = float)




    def CalculatePose(self, file_path):


        Final_3D_point, Final_2D_point = self.ExtractValuesFromYolo(file_path)# File pathen är faktiska pathen till där yolo .txt filen befinner sig
       


        if len(Final_2D_point) >= 4 and Final_3D_point is None: # Det var or här innan
            success, rvec, tvec, inliers = cv2.solvePnPRansac(Final_3D_point, Final_2D_point, self.camera_matrix, 
            self.dist_coeffs, iterationsCount = 100, reprojectionError = 8.0, flags = cv2.SOLVEPNP_ITERATIVE ) # Här utförs självaste uträkningen av rvec och tvec
        else:
            print("Not enough points detected to solve for pose.")


        return (rvec, tvec)if success and inliers is not None else (None, None)
   




def main(): # glöm ej att använda cv2.projectPoints för att verifiera rvec och tvec




        folder_path ="Ge folder vägen till där yolo datat finns"

        # Instantiate the processor
        processor = PnP_processing()


        if not os.path.exists(folder_path): # Till exempel "./yolo_results"
            print(f"Folder '{folder_path}' does not exist.")
            return

        for filename in os.listdir(folder_path):# filename är temporärt namn av yolo .txt filen, namnet är inte så viktigt, det viktiga är att det är en .txt fil
            if filename.endswith(".txt"): # Anledningen till varför man inte döper till riktiga filnamnet är för att man måste då byta filnman hela tiden efter man kör yolo modellen på nytt
                full_path = os.path.join(folder_path, filename)


                rvec, tvec = processor.CalculatePose(full_path)

                if rvec is not None and tvec is not None:
                    # Print the raw arrays
                    print("Rotation Vector (rvec):\n", rvec)# Angle in radians
                    print("Translation Vector (tvec):\n", tvec)
                    
                    # Print cleaner, one-line versions
                    print(f"Position (X, Y, Z): {tvec.flatten()}")

                

                

                #Här ska plotting ske

                img = np.zeros((800,1280,3), dtype=np.uint8)
                
                axis = np.float32([
                [0,0,0],
                [0.1,0,0],
                [0,0.1,0],
                [0,0,0.1]
                ])

                imgpts, jacobian = cv2.projectPoints(axis, rvec, tvec, processor.camera_matrix, processor.dist_coeffs)# cv2.projectPoints för att verifiera rvec och tvec
                
                imgpts = imgpts.astype(int)

                o, x, y, z = imgpts.reshape(-1,2)# x,y,z här är annorlunda från tvec x,y,z

                cv2.line(img, tuple(o), tuple(x), (0,0,255), 3) # X - Red
                cv2.line(img, tuple(o), tuple(y), (0,255,0), 3) # Y - Green
                cv2.line(img, tuple(o), tuple(z), (255,0,0), 3) # Z - Blue
                
                cv2_imshow("Pose estimation", img) #displaya bilden




                cv2.waitKey(0) #Kolla om detta behövs



        
        cv2.destroyAllWindows()


if __name__ == "__main__":
        main()




  