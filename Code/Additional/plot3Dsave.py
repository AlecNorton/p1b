import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from Code.Additional.DataReader import DataReader
from p1a.rotplot import rotplot
import argparse
import pathlib
from scipy.io import loadmat
from functools import partial 
import matplotlib.animation as animation
from rich.progress import Progress
# Initialize figure and axis
import matplotlib
import cv2
#matplotlib.use('Agg'

def main():
    parser = argparse.ArgumentParser(description="Compute and plot orientations using 4 methods for all available datasets.")
    parser.add_argument('--mat_number', type=int, required=False, default=1, help='train mat number (1..6)')
    parser.add_argument('--file_path', type = str, required = False, default = 'outputs', help = 'File directory where output .mat files are located')

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter()
    args = parser.parse_args()
    this_dir = pathlib.Path(__file__).resolve().parent
    output_dir = (this_dir / args.file_path).resolve()
    vid_dir = (output_dir / f"vid_{args.mat_number}").resolve()
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
    numFrames = len(mat['VICON']['ROLL'][0][0][0])
    #numFrames = 50
    vid_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, numFrames):
        for ax in axes:
            tx = ax.title.get_text()
            ax.clear()
            Z, Y, X = 0.0, 0.0, 0.0
            if('Gyroscope' in tx):
                Z = mat['GYRO']["YAW"][0][0][0][i]
                Y = mat['GYRO']["PITCH"][0][0][0][i]
                X = mat['GYRO']["ROLL"][0][0][0][i]
                tx = 'Gyroscope - ' + str(i)
            elif('Accelerometer'in tx):
                Z = mat['ACCEL']["YAW"][0][0][0][i]
                Y = mat['ACCEL']["PITCH"][0][0][0][i]
                X = mat['ACCEL']["ROLL"][0][0][0][i]
                tx = 'Accelerometer - ' + str(i)
    
            elif('Complementary Filter' in tx):
                Z = mat['COMP']["YAW"][0][0][0][i]
                Y = mat['COMP']["PITCH"][0][0][0][i]
                X = mat['COMP']["ROLL"][0][0][0][i]
                tx = 'Complementary Filter - ' + str(i)
            elif('Madgwick Filter' in tx):
                Z = mat['MADGWICK']["YAW"][0][0][0][i]
                Y = mat['MADGWICK']["PITCH"][0][0][0][i]
                X = mat['MADGWICK']["ROLL"][0][0][0][i]
                tx = 'Madgwick Filter - ' + str(i)
    
            elif('Vicon' in tx):
                Z = mat['VICON']["YAW"][0][0][0][i]
                Y = mat['VICON']["PITCH"][0][0][0][i]
                X = mat['VICON']["ROLL"][0][0][0][i]
                tx = 'Vicon - ' + str(i)
            else:
                print("error")
            #print("Z: ", Z)
            #print("Y: ", Y)
            rot = R.from_euler('ZYX', [Z, Y, X])
            rotplot(rot.as_matrix(), ax)
            ax.grid(True)
            ax.set_title(tx)
        plt.savefig(str(output_dir)+f"/vid_{args.mat_number}/{i}.png")
            
    #ani = FuncAnimation(fig, partial(update, axes = axes, mat = mat), frames = numFrames, interval = 1)
    #plt.show()
    #writerVid = animation.FFMpegWriter(fps = 60)
    #ani.save(str(output_dir)+f"/vid_{args.mat_number}.gif")
def update(frame, axes, mat):
    """Callback function that clears and replots each frame."""
    #print("Frame: " + str(frame))
    for ax in axes:
        tx = ax.title.get_text()
        ax.clear()
        Z, Y, X = 0.0, 0.0, 0.0
        if('Gyroscope' in tx):
            Z = mat['GYRO']["YAW"][0][0][0][frame]
            Y = mat['GYRO']["PITCH"][0][0][0][frame]
            X = mat['GYRO']["ROLL"][0][0][0][frame]
            tx = 'Gyroscope - ' + str(frame)
        elif('Accelerometer'in tx):
            Z = mat['ACCEL']["YAW"][0][0][0][frame]
            Y = mat['ACCEL']["PITCH"][0][0][0][frame]
            X = mat['ACCEL']["ROLL"][0][0][0][frame]
            tx = 'Accelerometer - ' + str(frame)

        elif('Complementary Filter' in tx):
            Z = mat['COMP']["YAW"][0][0][0][frame]
            Y = mat['COMP']["PITCH"][0][0][0][frame]
            X = mat['COMP']["ROLL"][0][0][0][frame]
            tx = 'Complementary Filter - ' + str(frame)
        elif('Madgwick Filter' in tx):
            Z = mat['MADGWICK']["YAW"][0][0][0][frame]
            Y = mat['MADGWICK']["PITCH"][0][0][0][frame]
            X = mat['MADGWICK']["ROLL"][0][0][0][frame]
            tx = 'Madgwick Filter - ' + str(frame)

        elif('Vicon' in tx):
            Z = mat['VICON']["YAW"][0][0][0][frame]
            Y = mat['VICON']["PITCH"][0][0][0][frame]
            X = mat['VICON']["ROLL"][0][0][0][frame]
            tx = 'Vicon - ' + str(frame)
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