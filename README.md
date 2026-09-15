# p1b
GitHub Project Repo for RBE 595-ST: Autonomous Drones with Group 3

# Instructions for Use

All code can be run in terminal with the command:

python wrapper.py

Running above command will perform the gyro, accelerometer, complementary filter, Madgwick filter, and the Unscented Kalman Filter on the first IMU trajectory data. 
Additional arguments can be made with --mat_number from 1-10, which will change the trajectory chosen for attitude estimation. --data_path, which is the location of the data folder respective to the python file. Folder chosen should have the following structure: data_folder/Data/Train/IMU/{list of .mat files corresponding to IMU}. Similarily under Train should be a Vicon folder which has a list of .mat files corresponding to the baseline Vicon motion capture. Under the Data folder should be an IMUParams mat. For mat_numbers above 6 (i.e. 7-10), an additional --test_path can be called that points to a folder containing IMU mat files. For instance:

python wrapper.py --mat_number 7 --test_path 'Test' 

will run the five attitude estimation filters on a file called IMURaw7.mat in a folder IMU in the folder Test. Subsequent results will be located in an outputs folder which contains plots and .mat files corresponding to estimated Roll, Pitch, and Yaw angles. 


