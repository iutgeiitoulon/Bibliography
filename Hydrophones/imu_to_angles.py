from datetime import datetime
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.optimize


parser = argparse.ArgumentParser(description='This script requires Numpy, Matplotlib, Scipy, and Pandas. \nThis script turns raw data of 9axis IMU to Yaw Pitch and Roll Angles. A calibration of the magnetometer can be done if calibration dataset is provided. Input dataset(s) must be in .csv file format with exactly those columns labelled in it : ["time","AccX","AccY","AccZ","GyrX","GyrY","GyrZ","MagX","MagY","MagZ"]. \nThe script saves angles in .csv format with one or two plotted figures (.png) showing the result without & with calibration of the magnetometer if it is done during the process. The saved files are located in a folder named "MPU_results\".')

parser.add_argument('--dataset', type=str, help='path of the .csv dataset file, with ["time","AccX","AccY","AccZ","GyrX","GyrY","GyrZ","MagX","MagY","MagZ"] columns')
parser.add_argument('--calibration', type=str,  help='path of the .csv CALIBRATION dataset file, with ["time","AccX","AccY","AccZ","GyrX","GyrY","GyrZ","MagX","MagY","MagZ"] columns. It should be the recording of the IMU data while tilting the board around every axis, and during a suficient amount of time to get lots of samples.')

args = parser.parse_args()
#print(args.accumulate(args.integers))

calib_dataset_path = args.calibration if args.calibration else False
dataset_path = args.dataset if args.dataset else False

if not calib_dataset_path :
    print("\nWARNING : Magnetometer won't be calibrated because calibration dataset filepath was not provided.\n")

def get_angles(data, output='degree') :

    Ax = data['AccX']
    Ay = data['AccY']
    Az = data['AccZ']
    Mx = data['MagX']
    My = data['MagY']
    Mz = data['MagZ']
    
    pitch = np.arctan2(Ax , np.sqrt(Ay**2 + Az**2))
    roll = np.arctan2(Ay , np.sqrt(Ax**2 + Az**2))
    
    magx = Mx*np.cos(pitch) + My*np.sin(roll)*np.sin(pitch) + Mz*np.cos(roll)*np.sin(pitch)
    magy = My*np.cos(roll) - Mz * np.sin(roll)
    
    yaw = np.arctan2(-magy , magx)
 
    out = np.array([yaw, pitch, roll]).transpose()
    
    if output == 'degree' :
        out = out * 180/np.pi
    
    return out

def ellipsoid_fit(X) :

    # need nine or more data points
    assert np.shape(X)[1] == 3
    x = X[:,0].reshape(-1,1)
    y = X[:,1].reshape(-1,1)
    z = X[:,2].reshape(-1,1)
    #print(x.shape)
    if np.shape(x)[0] < 9 :
        print( 'Must have at least 9 points to fit a unique ellipsoid' )

    D = np.concatenate((x**2, y**2, z**2, 
                2 * x*y, 2 * x*z, 2 * y*z,
                2 * x, 2 * y, 2 * z), axis = 1)
    # ndatapoints x 9 ellipsoid parameters

    #print(D)
    A=D.T @ D
    B=D.T @ np.ones((np.shape(x)[0],1))
    
    #print(A.shape)
    #print(B.shape)
    
    # solve the normal system of equations
    v = scipy.optimize.lsq_linear(A, B.flatten())
    #print(v)
    v = v.x.flatten()
    #v = np.linalg.lstsq(D.T @ D, D.T @ np.ones((np.shape(x)[0],1)) , rcond = None)
    #print(D.T @ D)
    #print(v)
    #print(v[7])
    # form the algebraic form of the ellipsoid
    A = np.array([[v[0], v[3], v[4], v[6]],
                [v[3], v[1], v[5], v[7]],
                [v[4], v[5], v[2], v[8]],
                [v[6], v[7], v[8], -1 ]])
    # find the center of the ellipsoid
    center = np.linalg.lstsq(-A[0:3, 0:3], np.array([v[6],v[7],v[8]]).reshape(-1,1), rcond = None)
    center = center[0].flatten()
    #print(center)
    # form the corresponding translation matrix
    T = np.eye( 4 )
    T[3, 0:3 ] = center
    #print(T)
    # translate to the center
    R = T @ A @ T.T
    #print(R)
    # solve the eigenproblem
    evals, evecs = np.linalg.eig(R[0:3, 0:3] / -R[3,3])
    
    #print(evals)
    #print(evecs)

    #radii = sqrt( 1 ./ diag( evals ) );
    evals=evals.reshape(-1,1)
    #print(1 / evals )
    #radii = np.sqrt( 1 / np.abs(evals) )
    radii = np.sqrt( 1 / evals)
    #print(radii)

    return center, radii, evecs, v


def mag_calibration(calib_dataframe, normalisation = True) :

    X = calib_dataframe[['MagX','MagY','MagZ']].values

    ## step 1 :
    ## estimation of the center of the ellipsoid and the magnetic field strength :
    precal_center, magfield = get_EllipsoidCenter_MagnFieldStr(X)
    #print(center)
    #print(magfield)
    
    ## step 2 :
    ## recenter mag data :
    X_centered =  X - precal_center

    ## step 3 :
    ## do ellipsoid fitting
    e_center, e_radii, e_eigenvecs, e_algebraic = ellipsoid_fit(X_centered)
    #print(e_center, e_radii, e_eigenvecs, e_algebraic)

    #print(e_center)
    #print(e_radii)
    #print(e_eigenvecs)
    #print(e_algebraic)


    ## step 4 :
    # compensate distorted magnetometer data
    # e_eigenvecs is an orthogonal matrix, so we can transpose instead of inversing it
    S = X_centered - e_center
    #print(S.shape)


    #scale = np.linalg.inv(np.array([[e_radii[0,0], 0, 0],
    #                                [0, e_radii[1,0], 0],
    #                                [0, 0, e_radii[2,0]]])) * np.min(e_radii) # scaling matrix
    
    scale = np.linalg.inv(np.array([[e_radii[0,0], 0, 0],
                                    [0, e_radii[1,0], 0],
                                    [0, 0, e_radii[2,0]]])) # scaling matrix

    #print(scale)
    maps = e_eigenvecs.transpose() # transformation matrix to map ellipsoid axes to coordinate system axes
    #print(maps)
    invmap = e_eigenvecs # inverse of above
    comp = invmap @ scale @ maps
    #print(comp)
    
    #print(Scomp.shape)
    
    #ellips_params_comp = ellipsoid_fit(Scomp)
    
    
    #if normalisation :
        #return Scomp
    
    #else :
        #return Scomp * magfield
    
    offset = precal_center + e_center
    return offset, comp
    
    
def compensate(data, comp_matrix, offset) :

    assert data.shape[1] == 3
    
    data = data - offset
    #print(data)
    data_comp = comp_matrix @ data.transpose() # do compensation
    #print(data_comp)
    data_comp = data_comp.transpose()
    #print(data_comp)
    
    return data_comp

def get_EllipsoidCenter_MagnFieldStr(MagValues) :
    # MagValues is a Nx3 array containing N x (MagX, MagY and MagZ) data triplets
    
    # returns a tuple (ellipsoid_center, Magnetic_Field_Strength)
    # ellipsoid_center : 1x3 array , the center of the fitted ellipsoid of the 
    # uncalibrated magnetometer data points

    
    def residual(p, x, y):
        res = []
        for i in range(x.shape[0]) :
            res.append(y[i,:] - np.dot(x[i,:],p))
        res = np.array(res).flatten()
        return res
    
    X = np.concatenate((MagValues,np.ones(MagValues.shape[0]).reshape(-1,1)),axis=1)
    Y = (X[:,0]**2+X[:,1]**2+X[:,2]**2).reshape(-1,1)
    p0 = np.array([1.0, 1.0, 1.0, 1.0])

    #print(X)
    #print(Y)
    #print(residual(p0, X, Y))

    popt, pcov = scipy.optimize.leastsq(residual, p0,  args=(X, Y))

    #print(popt)

    # center of ellipsoid
    V = 1/2 * popt[0:3]
    #print(V)
    # magnetic field strength :
    B = np.sqrt(popt[3] + np.dot(V,V))
    #print(B)
    return (V, B)
    
    
    
    
    
# set up
cwd = os.getcwd()
# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%Y%m%d_%Hh%Mm%Ss")
print("session time : ", dt_string,"\n")
# Directory
directory = "MPU_results"
# Path
savepath = os.path.join(cwd, directory)
# Create the output directory
os.makedirs(savepath, exist_ok = True)

print("folder \'MPU_results\' created in "+str(os.getcwd()))
print("output angles dataframes and figures will be saved there.")


# get IMU data
#%matplotlib inline

#baselink = "/nfs/NAS5/SABIOD/public_data/nthellier/QHB/MPU/data/"
#calib_dataset_path = baselink + "mpu_test1_ant1.csv"
labels = ["time","AccX","AccY","AccZ","GyrX","GyrY","GyrZ","MagX","MagY","MagZ"]
Magneto = ['MagX','MagY','MagZ']

#dataset_path = baselink + "mpu_test2_ant1.csv"
dataset = pd.read_csv(dataset_path, names=labels)



#calibration = False
calibration = calib_dataset_path


angles_notcalib = get_angles(dataset)
df_angles_notcalib = pd.DataFrame(angles_notcalib, columns = ['Yaw', 'Pitch', 'Roll'])
df_angles_notcalib.to_csv(os.path.join(savepath, dt_string+'_angles_mag_not_calib.csv')) 

fig = plt.figure(figsize=(10,4))
for i in range(angles_notcalib.shape[1]) :
    plt.scatter(dataset['time'], angles_notcalib[:,i],s=1)
plt.legend(["yaw","pitch","roll"])
plt.yticks(np.linspace(-180, 180, num=13, endpoint=True))
plt.grid('on')
plt.title('Angles without Magnetometer Calibration')
plt.savefig(os.path.join(savepath, dt_string+'_angles_mag_not_calib.png'))
plt.close()

if calibration :

    calib_df = pd.read_csv(calib_dataset_path, names=labels)
    #print(calib_df.head())
    offset, comp_matrix = mag_calibration(calib_df, normalisation = True)
    print("Magnetometer calibration results : \n")
    print("offset : ")
    print(offset)
    print("compensation Matrix : ")
    print(comp_matrix)

    calib_df_comp = calib_df.copy()
    calib_df_comp[Magneto] = compensate(calib_df_comp[Magneto].values, comp_matrix, offset)

    dataset[Magneto] = compensate(dataset[Magneto].values, comp_matrix, offset)
    
    angles = get_angles(dataset)
    
    fig = plt.figure(figsize=(10,4))
    for i in range(angles.shape[1]) :
        plt.scatter(dataset['time'], angles[:,i],s=1)
    plt.legend(["yaw","pitch","roll"])
    plt.yticks(np.linspace(-180, 180, num=13, endpoint=True))
    plt.grid('on')
    plt.title('Angles with Magnetometer Calibration')
    plt.savefig(os.path.join(savepath, dt_string+'_angles_mag_calib.png'))
    plt.close()

    df_angles = pd.DataFrame(angles, columns = ['Yaw', 'Pitch', 'Roll'])
    df_angles.to_csv(os.path.join(savepath, dt_string+'_angles_mag_calib.csv')) 
