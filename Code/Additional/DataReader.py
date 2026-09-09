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
                #Converting linear accelerations
                if(val_type_index < 3):
                    new_val = (float(val_instance)+float(self.params_mat['IMUParams'][1][val_type_index]))/float(self.params_mat['IMUParams'][0][val_type_index])
                else:
                    new_val = (3300/1023) * (math.pi/180) *.3 * (float(val_instance) - self.gyro_bias[val_type_index-3])
                #Converting rotational accelerations.
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
            gyro_bias_x = gyro_bias_x + int(self.imu_mat['vals'][3][i])
            gyro_bias_y = gyro_bias_y + int(self.imu_mat['vals'][4][i])
            gyro_bias_z = gyro_bias_z + int(self.imu_mat['vals'][5][i])
        gyro_bias = [gyro_bias_x, gyro_bias_y, gyro_bias_z]
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
                unit_start_accel = start_accel/np.linalg.norm(start_accel)
                unit_end_accel = end_accel/np.linalg.norm(end_accel)
                cosOmega = np.dot(unit_start_accel, unit_end_accel)
                #print(f"Omega: {math.acos(cosOmega)}")
                
                ratio = 1 - (step_ts - ts)/(step_ts - start_ts)
                #print(f"Ratio: {ratio}")
                begin_search_val = index
                if(math.acos(cosOmega) != 0):
                    new_accel = (math.sin((1-ratio)*math.acos(cosOmega))/math.sin(math.acos(cosOmega)))*unit_start_accel 
                    new_accel = new_accel + (math.sin(ratio*math.acos(cosOmega))/math.sin(math.acos(cosOmega)))*unit_end_accel
                    new_accel = new_accel * ((1-ratio)*np.linalg.norm(start_accel) + ratio*np.linalg.norm(end_accel))
                else:
                    new_accel = (1-ratio)*unit_start_accel + ratio*unit_end_accel
                    new_accel = new_accel * ((1-ratio)*np.linalg.norm(start_accel) + ratio*np.linalg.norm(end_accel))

                #print(f"New accel: {new_accel}")
                for i in range(0, 6):
                    new_imu_data[i].append(new_accel[i])
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
            
        

