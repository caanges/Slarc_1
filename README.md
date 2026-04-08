# Slarc_1

This project focuses on estimating the pose of an Unmanned Ground Vehicle (UGV) from an Unmanned Aerial Vehicle (UAV) using vision-based techniques.

Instead of relying on fiducial markers like AprilTag or ArUco, this approach explores natural landmark detection and feature-based methods, making the system more robust and deployable in real-world environments.

_______________________________________
## ⚙️ System Architecture

UAV (Camera Platform)

Captures top-down imagery
Detects UGV using:
Feature matching OR
Neural network

UGV (Target)

No markers required
Recognized via natural features

Pipeline

Image acquisition
Feature extraction / inference
Feature matching or detection
Pose estimation
Output relative pose (position + orientation)

______________________
## 🔬 Key Concepts

Feature detection & matching
Visual odometry
Perspective-n-Point (PnP)
End-to-end deep learning for localization

___________________________
## 🛠️ Setup & Installation

- git clone https://github.com/caanges/Slarc_1.git
- cd Slarc_1
- pip install -r requirements.txt
___________________
## 👥 Contributors

- Christoffer Angestam
- Emil Ekengren
- Elwin Green
- Edvin Mörk
- Malek Saleh


