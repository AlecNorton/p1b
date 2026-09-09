import scipy
from scipy import io
import math
import numpy as np
class DataReader:
    def __init__(self, mat_number):
        self.imu_mat_path = "p1a/Data/Train/IMU/imuRaw" + str(mat_number) + ".mat"
        self.vicon_mat_path = "p1a/Data/Train/Vicon/viconRot" + str(mat_number) + ".mat"
        self.imu_mat = io.loadmat(self.imu_mat_path)
        self.vicon_mat = io.loadmat(self.vicon_mat_path)
        self.params_mat = io.loadmat('p1a/IMUParams.mat')
        self.gyro_bias = self.gyro_bias_avg(100)
        self.clean()
        self.align()
    
    #Convert IMU linear and rotational velocities based on described equations. 
    def clean(self):
        #Cleaning IMU
        val_type_index = 0
        new_list = []
        for val_type in self.imu_mat['vals']:
            new_list.append([])
            for val_instance in val_type:
                # For accelerometer channels (0..2) keep raw sensor values.
                # Conversion to physical units (scale + bias + g) is done later
                # in `accel_lowpass_filter` according to the report formulas.
                if val_type_index < 3:
                    new_val = float(val_instance)
                else:
                    new_val = (3300.0/1023.0) * (math.pi/180.0) * 0.3 * (float(val_instance) - self.gyro_bias[val_type_index-3])
                # Rotational rates stored in rad/s, linear accelerations left raw.
                new_list[val_type_index].append(new_val)
            val_type_index = val_type_index + 1
        self.imu_mat['vals'] = new_list
        self.imu_mat['ts'] = self.imu_mat['ts'][0]

        #Several parts of VICON data has NANS! Resolve by simply excluding them from dataset. 
        vicon_ts = self.vicon_mat['ts'][0]
        vicon_data = []
        for i in range(0, len(vicon_ts)):
            rotMat = self.vicon_mat['rots'][0:, 0:, i]
            if(np.isnan(rotMat[0][0])):
                #Invalid. 
                vicon_ts = np.delete(vicon_ts, i)
            else:
                vicon_data.append(rotMat)
        self.vicon_mat['rots'] = vicon_data
        self.vicon_mat['ts'] = vicon_ts            

    #Determine gyro bias by averaging 
    def gyro_bias_avg(self, num_entries):
        gyro_bias_x = 0
        gyro_bias_y = 0
        gyro_bias_z = 0
        for i in range(0, num_entries):
            gyro_bias_x = gyro_bias_x + float(self.imu_mat['vals'][3][i])
            gyro_bias_y = gyro_bias_y + float(self.imu_mat['vals'][4][i])
            gyro_bias_z = gyro_bias_z + float(self.imu_mat['vals'][5][i])
        gyro_bias = [gyro_bias_x, gyro_bias_y, gyro_bias_z]
        #print(np.array(gyro_bias).shape)
        return [bias/num_entries for bias in gyro_bias]

    def align(self):
        vicon_ts = self.vicon_mat['ts']
        imu_ts = self.imu_mat['ts']
        #If vicon is longer, interpolate imu so that it matches in length.
        begin_search_val = 0
        new_imu_data = [[], [], [], [], [], []]
        new_imu_ts = []
        #Likely not this one...?
        if(len(vicon_ts) > len(imu_ts)):
            print("Interpolating imu...")
            for ts in vicon_ts:
                #print(f"Target ts: {ts}")
                index, imu_ts = self.search(imu_ts, ts)
                if(index == -1):
                    #Nothing was found. Do not add
                    break
                #print(f"Index: {index}")
                start_ts = imu_ts[0]
                step_ts = imu_ts[1]
                #print(f"Start TS: {start_ts}")
                #print(f"End_ts: {step_ts}")
                start_accel = self.get_sample(index+begin_search_val, True)
                end_accel = self.get_sample(index+begin_search_val + 1, True)
                #print(f"Start acceleration: {start_accel}")
                #print(f"end_accel: {end_accel}")
                ratio = 1 - (step_ts - ts)/(step_ts - start_ts)
                begin_search_val = index
                # Interpolate each IMU channel (linear interpolation) between start and end samples.
                # start_accel and end_accel are 6-element arrays: [ax,ay,az,wz,wx,wy]
                new_sample = (1.0 - ratio) * start_accel + ratio * end_accel
                for i in range(0, 6):
                    new_imu_data[i].append(new_sample[i])
                new_imu_ts.append(ts)
            self.imu_mat['vals'] = new_imu_data
            self.imu_mat['ts'] = new_imu_ts
            new_vicon_ts = list(filter(lambda x: x >=new_imu_ts[0] and x <= new_imu_ts[-1], self.vicon_mat['ts']))
            startIndex = np.where(self.imu_mat['ts'] == new_vicon_ts[0])[0][0]

            endIndex = np.where(self.imu_mat['ts'] == new_vicon_ts[-1])[0][0] +1
            new_vicon_mat = [scipy.spatial.transform.Rotation.from_matrix(self.get_sample(i, False)) for i in range(startIndex, endIndex)]
            self.vicon_mat['rots'] = new_vicon_mat
            self.vicon_mat['ts'] = new_vicon_ts
        else:
            print("Interpolating vicon.")
            rotMats = []
            for i in range(0, len(vicon_ts)):
                rotMat = self.get_sample(i, False)
                try:
                    scipy.spatial.transform.Rotation.from_matrix(rotMat)
                    rotMats.append(rotMat)
                except ValueError:
                    print(f"Matrix {i} is not a valid rotational matrix...{rotMat}")
                    #print(f"Determinant is {scipy.linalg.det(rotMat)}")
                    vicon_ts = np.delete(vicon_ts, i)
            key_rot = scipy.spatial.transform.Rotation.from_matrix(rotMats)
            slerp = scipy.spatial.transform.Slerp(vicon_ts, key_rot)
            imu_ts = list(filter(lambda x: x >= vicon_ts[0] and x<= vicon_ts[-1], imu_ts))
            self.imu_mat['ts'] = imu_ts
            vals = self.imu_mat['vals']
            for i in range(0, 6):
                vals[i] = vals[i][0:len(imu_ts)]
            self.imu_mat['vals'] = vals
            self.vicon_mat['rots'] = slerp(imu_ts)

    def search(self, list, val):
        correctIndex = -1
        for i in range(0, len(list)-1):
            if(val > list[i] and val < list[i+1]):
                correctIndex = i
                break
            elif(val == list[i]):
                correctIndex = i
                break
        return correctIndex, list[correctIndex:]

    def get_sample(self, index, data_flag):
        if(data_flag):
            #Collect imu_mat data
            vals = self.imu_mat['vals']
            return np.array([vals[0][index], vals[1][index], vals[2][index], vals[3][index], vals[4][index], vals[5][index]])
        else:
            rot = self.vicon_mat['rots'][index]
            return rot
            
        
import argparse
import pathlib
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def project_to_so3(matrix):
    """Return the closest proper 3-D rotation matrix."""
    U, _, Vt = np.linalg.svd(matrix)
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        U[:, -1] *= -1.0
        rotation = U @ Vt
    return rotation


def load_calibrate_and_sync(reader):
    """Load the original samples and put Vicon on the IMU timeline.

    DataReader is intentionally left unchanged.  This function reloads the
    original arrays because DataReader.align() trims timestamps without always
    applying the same starting index to the IMU values.  Keeping the original
    values and timestamps together fixes that mismatch.
    """
    imu = io.loadmat(reader.imu_mat_path)
    vicon = io.loadmat(reader.vicon_mat_path)

    raw = np.asarray(imu["vals"], dtype=float)
    t_imu = np.asarray(imu["ts"], dtype=float).ravel()
    t_vicon = np.asarray(vicon["ts"], dtype=float).ravel()
    vicon_matrices = np.moveaxis(
        np.asarray(vicon["rots"], dtype=float), 2, 0
    )

    # Remove a Vicon timestamp and its corresponding matrix together.
    valid_vicon = (
        np.isfinite(t_vicon)
        & np.isfinite(vicon_matrices).all(axis=(1, 2))
    )
    t_vicon = t_vicon[valid_vicon]
    vicon_matrices = vicon_matrices[valid_vicon]

    # Sort and remove duplicate Vicon timestamps because Slerp requires a
    # strictly increasing sequence.
    order = np.argsort(t_vicon)
    t_vicon = t_vicon[order]
    vicon_matrices = vicon_matrices[order]
    t_vicon, unique = np.unique(t_vicon, return_index=True)
    vicon_matrices = vicon_matrices[unique]
    vicon_matrices = np.stack([
        project_to_so3(matrix) for matrix in vicon_matrices
    ])

    # Select the IMU samples that lie inside the Vicon time range.  Crucially,
    # apply the same Boolean mask to both timestamps and sensor values.
    overlap = (
        np.isfinite(t_imu)
        & (t_imu >= t_vicon[0])
        & (t_imu <= t_vicon[-1])
    )
    t_imu = t_imu[overlap]
    raw = raw[:, overlap]

    # Interpolate ground-truth rotations at the exact IMU timestamps.
    vicon_rotation = Slerp(
        t_vicon, R.from_matrix(vicon_matrices)
    )(t_imu)

    params = np.asarray(reader.params_mat["IMUParams"], dtype=float)
    scale = params[0]
    accel_bias = params[1]

    # Report equation: a_tilde = (a_raw * scale + bias) * 9.81.
    '''
    acceleration = (
        raw[0:3] * scale[:, None] + accel_bias[:, None]
    ) * 9.81
    '''
    ac_with_bias= (raw[0:3] * scale[:, None] + accel_bias[:, None])
    multiplier=9.81/(np.mean(np.sqrt(ac_with_bias[0,:100]**2 + ac_with_bias[1,:100]**2 + ac_with_bias[2,:100]**2)))
    acceleration=ac_with_bias * multiplier


    # Raw channel order is [ax, ay, az, wz, wx, wy].
    # Bias is calculated in the same raw order from the first 100 samples,
    # matching the unchanged DataReader implementation.
    gyro_bias = np.asarray(reader.gyro_bias, dtype=float)
    gyro_factor = (3300.0 / 1023.0) * (np.pi / 180.0) * 0.3
    gyro_raw_order = gyro_factor * (
        raw[3:6] - gyro_bias[:, None]
    )

    ax, ay, az = acceleration
    wz, wx, wy = gyro_raw_order
    return t_imu, ax, ay, az, wx, wy, wz, vicon_rotation


def integrate_gyro(ts, wx, wy, wz, R0_3x3):
    ts = np.asarray(ts).ravel()
    wx = np.asarray(wx).ravel(); wy = np.asarray(wy).ravel(); wz = np.asarray(wz).ravel()
    N = ts.size
    if any(arr.size != N for arr in [wx, wy, wz]):
        raise ValueError("ts,wx,wy,wz mismatch")
    matrices = np.empty((N, 3, 3), dtype=float)
    current = R.from_matrix(R0_3x3)

    matrices[0] = current.as_matrix()
    for k in range(N - 1):
        dt = ts[k + 1] - ts[k]
        if dt <= 0:
            raise ValueError("IMU timestamps must be strictly increasing")

        # Body-frame angular displacement during this sample interval.
        dR = R.from_rotvec(
            np.array([wx[k], wy[k], wz[k]]) * dt
        )
        current = current * dR
        matrices[k + 1] = current.as_matrix()
    return R.from_matrix(matrices)


def accel_tilt_from_calibrated(ax, ay, az):
    """Accelerometer attitude equations used by the supplied report."""
    roll = np.arctan2(ay, np.sqrt(ax**2 + az**2))
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    yaw = np.arctan2(np.sqrt(ax**2 + ay**2), az)
    return roll, pitch, yaw


def lowpass(values, alpha=0.8):
    """Report equation: x_hat[k]=(1-alpha)x[k]+alpha*x_hat[k-1]."""
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    result[0] = values[0]
    for k in range(1, len(values)):
        result[k] = (1.0 - alpha) * values[k] + alpha * result[k - 1]
    return result

def complimentary(ts, wx, wy, wz, R0_3x3,accel):
    alpha_accel=0.8
    alpha_comp=0.98
    ts = np.asarray(ts).ravel()
    wx = np.asarray(wx).ravel(); wy = np.asarray(wy).ravel(); wz = np.asarray(wz).ravel()

    N = ts.size
    if any(arr.size != N for arr in [wx, wy, wz]):
        raise ValueError("ts,wx,wy,wz mismatch")
    comp_rpy =np.empty((N,3), dtype=float)
    current_R = R.from_matrix(R0_3x3)
    current_rpy=current_R.as_euler('ZYX', degrees=False)
    
    comp_rpy[0]=current_rpy 
    for k in range(N-1):
        dt = ts[k + 1] - ts[k]
        if dt <= 0:
            raise ValueError("IMU timestamps must be strictly increasing")
        dR_gyro = R.from_rotvec(
            np.array([wx[k], wy[k], wz[k]]) * dt
        )
        gyro_update = current_R * dR_gyro # new gyro state
        gyro_update_rpy=gyro_update.as_euler('ZYX', degrees=False)

        accel_update_rpy = (1.0 - alpha_accel) * accel[:,k].T + alpha_accel*(current_rpy) # new accel state

        current_rpy = (1.0-alpha_comp)*accel_update_rpy+ alpha_comp*gyro_update_rpy #weighted sum in rpy space
        comp_rpy[k + 1,:] = current_rpy
        current_R = R.from_euler('ZYX', current_rpy)

        #print(accel_update_rpy)
        #print(gyro_update_rpy)
        #print(current_rpy)
        #print(current)
   
    return comp_rpy

def madgwick(ts, wx, wy, wz, R0_3x3,accel):
    beta=0.02
    ts = np.asarray(ts).ravel()
    wx = np.asarray(wx).ravel(); wy = np.asarray(wy).ravel(); wz = np.asarray(wz).ravel()

    N = ts.size
    if any(arr.size != N for arr in [wx, wy, wz]):
        raise ValueError("ts,wx,wy,wz mismatch")
    madgwick_rpy =np.empty((N,3), dtype=float)
    current_R = R.from_matrix(R0_3x3)
    current_rpy=current_R.as_euler('ZYX', degrees=False)
    q2,q3,q4,q1=current_R.as_quat() # qx qy qz w but following w,x,y,z convention
    current_q=np.array([q1, q2, q3, q4]).T #4,
    accel=accel/np.linalg.norm(accel,axis=0)
    ax,ay,az=accel
    madgwick_rpy[0]=current_rpy 
    for k in range(N-1):
        func=np.array([[2*(q2*q4-q1*q3)-ax[k]],
                       [2*(q1*q2+q3*q4)-ay[k]],
                       [2*(0.5-q2**2-q3**2)-az[k]]]) # q*g*q'-a basically cost function to minimize error
        jacob=np.array([[-2*q3, 2*q4, -2*q1, 2*q2],
                       [2*q2, 2*q1, 2*q4, 2*q3],
                        [0, -4*q2, -4*q3, 0]]) # Jacobian
        #print((jacob.T).shape)
        #rint(func.shape)
        func=np.reshape(func,(3,1))
        grad=jacob.T@func
        grad_norm=grad/(np.linalg.norm(grad)) # direction of norm
#________ ends accel contribution     

        d_gyro =np.array([wx[k], wy[k], wz[k],0])
        x1,y1,z1,w1=current_R.as_quat()
        lq=np.array([[ w1,  -z1, y1,  x1],
                    [z1,  w1,  -x1,  y1],
                    [ -y1, x1,  w1,  z1],
                    [-x1, -y1, -z1,  w1]])
        '''
        lq=np.array([[ w1,  z1, -y1,  x1],
                            [-z1,  w1,  x1,  y1],
                            [ y1, -x1,  w1,  z1],
                            [-x1, -y1, -z1,  w1]])'''
        wglobal = 0.5*(lq @ d_gyro) # w in global reference frame
        qx,qy,qz,qw=wglobal
        wglobal_q=np.array([qw,qx,qy,qz])
        fusion=wglobal_q[:,np.newaxis] - beta*grad_norm # 4*1 => one step descend
        dt = ts[k + 1] - ts[k]
        if dt <= 0:
            raise ValueError("IMU timestamps must be strictly increasing")
             
        fusion_int_rawq=current_q[:,np.newaxis] + fusion*dt # theta+w*dt = integration
        current_q=(fusion_int_rawq/np.linalg.norm(fusion_int_rawq)).ravel()
        q1,q2,q3,q4=current_q # w,x,y,z
        current_R=R.from_quat(np.array([q2,q3,q4,q1])) #x,y,z,w
  
        madgwick_rpy[k + 1,:] =  current_R.as_euler('ZYX', degrees=False)
   
    return madgwick_rpy



def main():
    parser = argparse.ArgumentParser(description="Compute and plot gyro-only and accel-only orientations from dataset number.")
    parser.add_argument('--mat_number', type=int, required=False, default=1, help='train mat number (1..6)')
    args = parser.parse_args()

    this_dir = pathlib.Path(__file__).resolve().parent
    output_dir = (this_dir / "outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = DataReader(args.mat_number)
    (
        t_imu, ax, ay, az, wx, wy, wz, vicon_rotation
    ) = load_calibrate_and_sync(reader)

    # Gyroscope-only attitude starts at the first synchronized Vicon pose.
    gyro_rotation = integrate_gyro(
        t_imu, wx, wy, wz, vicon_rotation[0].as_matrix()
    )

    gyro_zyx = gyro_rotation.as_euler("ZYX")
    vicon_zyx = vicon_rotation.as_euler("ZYX")

    gyro_yaw, gyro_pitch, gyro_roll = gyro_zyx.T

    # SciPy expresses Euler angles on their principal branches, so an angle
    # jumps from +pi to -pi when it crosses the branch boundary.  The report
    # displays the integrated gyro angles continuously instead.  Unwrapping
    # changes only that numerical representation; it does not change the
    # underlying orientation.  Keep Vicon wrapped to match the report plots.
    gyro_roll = np.unwrap(gyro_roll)
    gyro_pitch = np.unwrap(gyro_pitch)
    gyro_yaw = np.unwrap(gyro_yaw)
    vicon_yaw, vicon_pitch, vicon_roll = vicon_zyx.T

    accel_roll, accel_pitch, accel_yaw = (
        accel_tilt_from_calibrated(ax, ay, az)
    )
    accel_updates =np.stack([accel_yaw[:,None], accel_pitch[:,None], accel_roll[:,None]],axis=0)
    accel_roll = lowpass(accel_roll, alpha=0.8)
    accel_pitch = lowpass(accel_pitch, alpha=0.8)
    accel_yaw = lowpass(accel_yaw, alpha=0.8)
    
    
    #complimentary filter
    comp_rpy=complimentary(t_imu, wx, wy, wz, vicon_rotation[0].as_matrix(),accel_updates)

    accel_raw=np.stack([ax,ay,az],axis=0) 
    madgwick_rpy=madgwick(t_imu, wx, wy, wz, vicon_rotation[0].as_matrix(),accel_raw)

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    series = [
        ("Roll", vicon_roll, gyro_roll, accel_roll,comp_rpy[:,2],madgwick_rpy[:,2]),
        ("Pitch", vicon_pitch, gyro_pitch, accel_pitch,comp_rpy[:,1],madgwick_rpy[:,1]),
        ("Yaw", vicon_yaw, gyro_yaw, accel_yaw,comp_rpy[:,0],madgwick_rpy[:,0]),
    ]

    for axis, (name, vicon_angle, gyro_angle, accel_angle,comp_angle,mad_angle) in zip(axes, series):
        axis.plot(t_imu, vicon_angle, label="Vicon")
        axis.plot(t_imu, gyro_angle, label="Gyro-only")
        axis.plot(t_imu, accel_angle, label="Accel-only (LPF)")
        axis.plot(t_imu, comp_angle, label="Complimentary filter")
        axis.plot(t_imu, mad_angle, label="Madgwick filter")
        axis.set_ylabel(f"{name} (rad)")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(
        f"Attitude Comparison (mat {args.mat_number}): "
        "Gyro-only vs Accel-only"
    )
    figure.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_path = (
        output_dir / f"attitude_mat{args.mat_number}_gyro_accel_corrected.png"
    )
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"Synchronized samples: {len(t_imu)}")
    print(f"Accelerometer magnitude at rest: {np.mean(np.sqrt(ax[:100]**2 + ay[:100]**2 + az[:100]**2)):.3f} m/s^2")
    print(f"Figure saved to: {output_path}")


if __name__ == '__main__':
    main()
