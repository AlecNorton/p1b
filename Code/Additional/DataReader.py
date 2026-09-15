import scipy
from scipy import io
import math
import numpy as np
class DataReader:
    def __init__(self, mat_number, path_str, test = False, test_str = 'Test'):
        if(test == True):
            self.imu_mat_path = test_str + f"/IMU/imuRaw{mat_number}.mat"
            self.imu_mat = io.loadmat(self.imu_mat_path)
            self.vicon_mat = None
            #Still perform interpolation of Vicon just so we don't need to adjust further code. 
            self.params_mat = io.loadmat(path_str+'/IMUParams.mat')['IMUParams']
            self.gyro_bias = self.gyro_bias_calibrate(200)
            self.gyro_noise = self.gyro_noise_calibrate(200)
            self.clean()
            #self.align()
        else:
            self.imu_mat_path = path_str+ "/Data/Train/IMU/imuRaw" + str(mat_number) + ".mat"
            self.vicon_mat_path = path_str+ "/Data/Train/Vicon/viconRot" + str(mat_number) + ".mat"
            self.imu_mat = io.loadmat(self.imu_mat_path)
            self.vicon_mat = io.loadmat(self.vicon_mat_path)
            self.params_mat = io.loadmat(path_str+'/IMUParams.mat')['IMUParams']
            self.gyro_bias = self.gyro_bias_calibrate(200)
            self.gyro_noise = self.gyro_noise_calibrate(200)
            self.clean()
            self.align()

    
    #Convert IMU linear and rotational velocities based on described equations. 
    
    def clean(self):
        #Cleaning IMU
        
        self.imu_mat['ts'] = self.imu_mat['ts'][0]
        if(self.vicon_mat is None):
            pass
        else:
        #Several parts of VICON data has NANS! Resolve by simply excluding them from dataset. 
            vicon_ts = self.vicon_mat['ts'][0]
            vicon_data = []
            rotMats = []
            for i in range(0, len(vicon_ts)):
                rotMat = self.vicon_mat['rots'][0:, 0:, i]
                if(np.isnan(rotMat[0][0])):
                    #Invalid. 
                    vicon_ts = np.delete(vicon_ts, i)
                    print(f"Matrix {i} included NaNs, do not include...")
                    continue
                try:
                    scipy.spatial.transform.Rotation.from_matrix(rotMat)
                    rotMats.append(rotMat)
                except ValueError:
                    print(f"Matrix {i} is not a valid rotational matrix...{rotMat}")
                    #print(f"Determinant is {scipy.linalg.det(rotMat)}")
                    vicon_ts = np.delete(vicon_ts, i)
            self.vicon_mat['rots'] = scipy.spatial.transform.Rotation.from_matrix(rotMats)
            self.vicon_mat['ts'] = vicon_ts     
      

    #Determine gyro bias by averaging 
    def gyro_bias_calibrate(self, num_entries):
        gyro_bias_z = np.mean(self.imu_mat['vals'][3][0:num_entries])
        gyro_bias_x = np.mean(self.imu_mat['vals'][4][0:num_entries])
        gyro_bias_y = np.mean(self.imu_mat['vals'][5][0:num_entries])

        return [gyro_bias_z, gyro_bias_x, gyro_bias_y]

    def gyro_noise_calibrate(self, num_entries):
        gyro_noise = []
        for gyro in range(3, 6):
            new_list = []
            for i in range(0, len(self.imu_mat['vals'][gyro])):
                vals = self.get_sample(i, True)
                new_list.append(vals[gyro])
            gyro_noise.append(np.std(new_list))
        return gyro_noise

    def accel_model_calibrate(self, num_entries):
        accel_bias_x = np.mean(self.imu_mat['vals'][0][0:num_entries])
        accel_bias_y = np.mean(self.imu_mat['vals'][1][0:num_entries])
        accel_bias_z = np.mean(self.imu_mat['vals'][2][0:num_entries])

        return [[accel_bias_x, accel_bias_y, accel_bias_z]]

    def align(self):
        vicon_ts = self.vicon_mat['ts']
        imu_ts = self.imu_mat['ts']
        #If vicon is longer, interpolate imu so that it matches in length.
        begin_search_val = 0
        new_imu_data = [[], [], [], [], [], []]
        new_imu_ts = []

        print("Interpolating vicon.")
        slerp = scipy.spatial.transform.Slerp(vicon_ts, self.vicon_mat['rots'])
        #Ensure all imu_ts are within slerp ability to inteprolate. 
        new_imu_ts = list(filter(lambda x: x >= vicon_ts[0] and x<= vicon_ts[-1], imu_ts))
        #Get start and end index of new range of new_imu_ts
        startIndex = np.where(self.imu_mat['ts'] == new_imu_ts[0])[0][0]
        endIndex = np.where(self.imu_mat['ts'] == new_imu_ts[-1])[0][0]+1
        self.imu_mat['ts'] = new_imu_ts
        self.vicon_mat['ts'] = new_imu_ts
        vals = self.imu_mat['vals']
        new_list = []
        for i in range(0, 6):
            new_list.append(vals[i][startIndex:endIndex])
        self.imu_mat['vals'] = new_list
        self.vicon_mat['rots'] = slerp(new_imu_ts)

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

    def get_sample(self, i, data_flag):
        if(data_flag):
            #Collect imu_mat data
            vals = self.imu_mat['vals']
            return np.array(self.convert_linear_accel([vals[0][i], vals[1][i], vals[2][i]])+self.convert_rotational_accel([vals[3][i], vals[4][i], vals[5][i]]))
        else:
            rot = self.vicon_mat['rots'][i]
            return rot
    
    def convert_linear_accel(self, accel):
        
        [ax, ay, az] = [accel[0], accel[1], accel[2]]
        '''
        conv_ax = ((ax + self.params_mat[1][0]) / self.params_mat[0][0])
        conv_ay = ((ay + self.params_mat[1][1]) / self.params_mat[0][1])
        conv_az = ((az + self.params_mat[1][2]) / self.params_mat[0][2])
        '''
        
        conv_ax = (ax*self.params_mat[0][0] + self.params_mat[1][0]) *9.81
        conv_ay = (ay*self.params_mat[0][1] + self.params_mat[1][1])*9.81
        conv_az = (az*self.params_mat[0][2] + self.params_mat[1][2])*9.81
    
        return [conv_ax, conv_ay, conv_az]

    def get_all_imu_samples(self):
        axs, ays, azs, wzs, wys, wxs = [], [], [], [], [], []
        for i in range(len(self.imu_mat['ts'])):
            ax, ay, az, wz, wy, wx =  self.get_sample(i, True)
            axs.append(ax)
            ays.append(ay)
            azs.append(az)
            wxs.append(wx)
            wys.append(wy)
            wzs.append(wz)
        return axs, ays, azs, wzs, wys, wxs
    
    def convert_rotational_accel(self, acc):
        [wz, wx, wy] = [acc[0], acc[1], acc[2]]
        conv_wx = (3300/1023)* (math.pi/180) * .3 * (wx - self.gyro_bias[1])
        conv_wy = (3300/1023)* (math.pi/180) * .3 * (wy - self.gyro_bias[2])
        conv_wz = (3300/1023)* (math.pi/180) * .3 * (wz - self.gyro_bias[0])
        return [conv_wz, conv_wy, conv_wx]
            
        
            
        

