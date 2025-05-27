# Automated-Data-Generation-for-Pose-Estimation
Traditional methods of pose estimation of objects include preparing a dataset specific to the object, training the data so the trained AI model is able to identify the object from the infrastructure cameras. However collection or generation of synthetic data includes manual intervention, making it more time consuming

* Step 1 -
Download the Backgrounds folder and use it as different textures for rendering. The images can be taken from official COCO Dataset website too

* Step 2 -
Prepare your CAD model in .glb format. Make sure the colors are visible, if not use the following document to bake colors using tools like Blender. Scale your CAD model according to actual measurements.

* Step 3 -
Use the sample code to test how your CAD model looks in the trimesh scene

* Step 4 -
Run data_gen3.py in order to generate the complete synthetic dataset. The dataset covers nearly 10GB of space. Make sure you don't run out of space.

* Step 5 -
To verify the data use overlay.py to make sure that the bounding boxes and the keypoints are intact on the object.

* Step 6 -
Run keypoint_extraction.py and generate a text file containing pre-defined 3D keypoints on the object. Save them for future use

* Step 7 -
Train the YOLOv8 pose estimation model. Use the train.py script

* Step 8 -
Compute your camera intrinsics, homography and store them. Use the computed calibration parameters in csvwriter.py. Also add in the pre-defined keypoints to this script.
Run this script and the 2D poses, x,y and yaw are generated in a csv file.
