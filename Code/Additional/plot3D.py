import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from DataReader import DataReader
from rotplot import rotplot
import argparse
import pathlib
from scipy.io import loadmat
from functools import partial 
import matplotlib.animation as animation
from rich.progress import Progress
# Initialize figure and axis
import matplotlib
#matplotlib.use('Agg')

sample = 5
def main():
    parser = argparse.ArgumentParser(description="Compute and plot orientations using 4 methods for all available datasets.")
    parser.add_argument('--mat_number', type=int, required=False, default=1, help='train mat number (1..6)')
    parser.add_argument('--file_path', type = str, required = False, default = 'outputs', help = 'File directory where output .mat files are located')


    args = parser.parse_args()
    this_dir = pathlib.Path(__file__).resolve().parent
    output_dir = (this_dir / args.file_path).resolve()
    mat = loadmat(str(output_dir)+f"/attitude_{args.mat_number}")
    fig = plt.figure( figsize=(16, 9))
    gyro = fig.add_subplot(1, 5, 1, projection='3d')
    gyro.set_title('Gyroscope')
    accel = fig.add_subplot(1, 5, 2, projection = '3d')
    accel.set_title('Accelerometer')
    comp = fig.add_subplot(1, 5, 3, projection = '3d')
    comp.set_title('Complementary Filter')
    madgwick = fig.add_subplot(1, 5, 4, projection = '3d')
    madgwick.set_title('Madgwick Filter')
    vicon = fig.add_subplot(1, 5, 5, projection = '3d')
    vicon.set_title('Vicon')
    axes = [gyro, accel, comp, madgwick, vicon]
    numFrames = len(mat['VICON']['Roll'][0][0][0])
    #numFrames = 50
    ani = FuncAnimation(fig, partial(update, axes = axes, mat = mat), frames = numFrames//sample, interval = 1)
    #plt.show()
    writerVid = animation.FFMpegWriter(fps = 60)
    ani.save(str(output_dir)+f"/vid_{args.mat_number}.gif")
def update(frame, axes, mat):
    """Callback function that clears and replots each frame."""
    #print("Frame: " + str(frame))
    for ax in axes:
        tx = ax.title.get_text()
        ax.clear()
        Z, Y, X = 0.0, 0.0, 0.0
        if('Gyroscope' in tx):
            Z = mat['GYRO']["Yaw"][0][0][0][frame*sample]
            Y = mat['GYRO']["Pitch"][0][0][0][frame*sample]
            X = mat['GYRO']["Roll"][0][0][0][frame*sample]
            tx = 'Gyroscope - ' + str(frame*sample)
        elif('Accelerometer'in tx):
            Z = mat['ACCEL']["Yaw"][0][0][0][frame*sample]
            Y = mat['ACCEL']["Pitch"][0][0][0][frame*sample]
            X = mat['ACCEL']["Roll"][0][0][0][frame*sample]
            tx = 'Accelerometer - ' + str(frame*sample)

        elif('Complementary Filter' in tx):
            Z = mat['COMP']["Yaw"][0][0][0][frame*sample]
            Y = mat['COMP']["Pitch"][0][0][0][frame*sample]
            X = mat['COMP']["Roll"][0][0][0][frame*sample]
            tx = 'Complementary Filter - ' + str(frame*sample)
        elif('Madgwick Filter' in tx):
            Z = mat['MADGWICK']["Yaw"][0][0][0][frame*sample]
            Y = mat['MADGWICK']["Pitch"][0][0][0][frame*sample]
            X = mat['MADGWICK']["Roll"][0][0][0][frame*sample]
            tx = 'Madgwick Filter - ' + str(frame*sample)

        elif('Vicon' in tx):
            Z = mat['VICON']["Yaw"][0][0][0][frame*sample]
            Y = mat['VICON']["Pitch"][0][0][0][frame*sample]
            X = mat['VICON']["Roll"][0][0][0][frame*sample]
            tx = 'Vicon - ' + str(frame*sample)
        else:
            print("error")
        #print("Z: ", Z)
        #print("Y: ", Y)
        rot = R.from_euler('ZYX', [Z, Y, X])
        rotplot(rot.as_matrix(), ax)
        ax.grid(True)
        ax.set_title(tx)

    #print("Update one frame.")



# Create the animation
# frames: total number of frames
# interval: delay between frames in milliseconds

main()