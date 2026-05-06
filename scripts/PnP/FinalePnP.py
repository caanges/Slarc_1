import cv2
import numpy as np
import math
import os





class PnP_processing:


    def __init__(self):


        UGV_points_3D = np.array([
            [5.25, 40.4, 4.5],
            [-9.75, 29.0, -2.5],
            [9.75, 29.0, -2.5],
            [22.5, 16.25, 0.0],
            [-22.5, 16.25, 0.0],
            [22.5, -16.25, 0.0],
            [15.5, -11.75, 6.0],
            [-15.5, -11.75, 6.0],
            [-22.5, -16.25, 0.0],
            [9.75, -29.25, -2.5],
            [-9.75, -29.25, -2.5],
            [8.5, -43.25, -5.0],
            [-8.5, -43.25, -5.0]
        ] ,dtype = np.float32)


        focal_length = 640 
        center = (640 / 2, 400 / 2) 
        self.camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
        ], dtype=np.float32)


        self.dist_coeffs = np.zeros((4, 1))


        self.conf_threshold = 0.8


        self.img_w = 640 
        self.img_h = 400


    def ExtractValuesFromYolo(self, file_path):
        # Reads the .txt file that yolo has stored all data in and converts it to a list for pnp
        Valid_2D_image_points = []
        matching_3D_object_points = []


        if not os.path.exists(file_path):
            return None, None
       
        with open(file_path, "r") as file:
            raw_data = file.read().split() 
            # enumerate(file, start=1) gives us (1, line1), (2, line2), etc. Vi gör detta för att skapa class_ids för alla data punkter från 1-13 

        # 2. "Chunk" the flat list into rows. 
        # Assuming your real inference data has 6 values per point: (class, x, y, w, h, conf)
        values_per_row = 6
        rows = [raw_data[i : i + values_per_row] for i in range(0, len(raw_data), values_per_row)]# Rows är en lista av listor som har ordnats upp så att varje keypoint reprsenterar en lista med dess respektive egan specifika värden

        for row_idx, line in enumerate(rows, start =0): # Att det står start = 1, tror jag betyder att class_id börjar från 1 och inte 0. Kanske börja från 0 istället
            # Assumed that YOLO format: class_id x_center y_center width height confidence
            #parts = line.split()
            if len(line) < values_per_row: continue # Skip broken lines # Förrut var det < 6, Det brukade stå if len(parts)


            #class_id = int(parts[0]) # I denna for loop så läses det in en linje för sig i .txt filen och därav så tar man fram dess class_id, conf, x, y och om dess confidence är tillräcklig så kan den skickas in i accepterade 2d punkter, och matchande 3d punkten registreras då också med samma class_id
            # Denna class_id variabel hållaren försvann då vi rangordnar utifrån raderna som man kan se ovan, och inte via parts[0], då det alltid siffrorna 0,1, och 2 alltid.
            # Convert normalized (0-1) to pixels
            x_pixel = float(parts[1]) * self.img_w
            y_pixel = float(parts[2]) * self.img_h
            conf = float(parts[5])


            if conf >= self.conf_threshold:
                # row_idx 1 matches self.UGV_points_3D[1], row_idx 2 matches [2], etc.
                if row_idx < len(self.UGV_points_3D):# Denna säger följande: Only proceed with the calculation if the current row number corresponds to a point that actually exists in our 3D array.
                    Valid_2D_image_points.append([x_pixel, y_pixel])
                    matching_3D_object_points.append(self.UGV_points_3D[row_idx])


        return np.array(matching_3D_object_points, dtype = np.float32), np.array(Valid_2D_image_points, dtype = float)




    def CalculatePose(self, file_path):

        success = False #initialisering
        rvec, tvec, inliers = None, None, None



        Final_3D_point, Final_2D_point = self.ExtractValuesFromYolo(file_path)# File pathen är faktiska pathen till där yolo .txt filen befinner sig
       


        if len(Final_2D_point) >= 4 and Final_3D_point is not None: # Det var or här innan #Ska det inte finnas final_3d_point is not None?
            success, rvec, tvec, inliers = cv2.solvePnPRansac(Final_3D_point, Final_2D_point, self.camera_matrix, 
            self.dist_coeffs, iterationsCount = 100, reprojectionError = 8.0, flags = cv2.SOLVEPNP_ITERATIVE ) # Här utförs självaste uträkningen av rvec och tvec
        else:
            print("Not enough points detected to solve for pose.")


        return (rvec, tvec)if success and inliers is not None else (None, None)
   




def main(): 


        

        folder_path = r"C:\Users\msh23003\OneDrive - Mälardalens universitet\Documents\txtFileForTestPnP"
        image_folder = r"C:\Users\msh23003\OneDrive - Mälardalens universitet\Documents\txtFileForTestPnP"

        # Instantiate the processor
        processor = PnP_processing()


        if not os.path.exists(folder_path): # Till exempel "./yolo_results"
            print(f"Folder '{folder_path}' does not exist.")
            return

        for filename in os.listdir(folder_path):# filename är temporärt namn av yolo .txt filen, namnet är inte så viktigt, det viktiga är att det är en .txt fil
            if filename.endswith(".txt"): # Anledningen till varför man inte döper till riktiga filnamnet är för att man måste då byta filnman hela tiden efter man kör yolo modellen på nytt
                full_path = os.path.join(folder_path, filename)
                base_name = os.path.splitext(filename)[0]

                img_path =os.path.join(image_folder, f"{base_name}.png")

                rvec, tvec = processor.CalculatePose(full_path)

                if rvec is not None and tvec is not None:
                    # Print the raw arrays
                    print("Rotation Vector (rvec):\n", rvec)# Angle in radians
                    print("Translation Vector (tvec):\n", tvec)
                    
                    # Print cleaner, one-line versions
                    print(f"Position (X, Y, Z): {tvec.flatten()}")

                
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                else:
                    print(f"No corresponding image found for {filename}. Using blank background")
                    img = np.zeros((800,1280,3), dtype=np.uint8)

                #Här ska plotting ske

                
                
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
                
                cv2.imshow("Pose estimation", img) #displaya bilden, Innan stod det cv2_imshow med udnerstreck vilket kommer från google collab



                '''
                # Display the plotted image
                try:
                    from google.colab.patches import cv2_imshow
                    cv2_imshow(img)
                except ImportError:
                    cv2.imshow("Pose estimation", img)
                    cv2.waitKey(0)
                '''


                cv2.waitKey(0) #Kolla om detta behövs



        
        cv2.destroyAllWindows()


if __name__ == "__main__":
        main()




  