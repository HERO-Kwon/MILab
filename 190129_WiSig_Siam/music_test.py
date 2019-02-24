import numpy as np
import pandas as pd
import os
import gzip,pickle
import math
import cmath
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

path_csi = 'G:\\Data\\Wi-Fi_processed\\'
path_meta = 'D:\\Data\\Wi-Fi_meta\\'
file = os.listdir(path_csi)[0]

# freq BW list
bw_list = pd.read_csv(path_meta+'wifi_f_bw_list_v1.csv')

# parameters
# parameters
d = 0.45
c = 300 * 10**6
ch=10
f_d = 20 * 10**6 / 30


def Sigma(theta,d,f,c):
    return cmath.exp(-1j * 2 * math.pi * d * math.sin(theta) * f / c)
def Omega(tau,f_d):
    return cmath.exp(-1j * 2 * math.pi * f_d * tau)
def SteerVec(theta,tau):
    list_a = []
    for rx in range(2):
        for subc in range(15):
            f = bw_list[str(subc)][ch-1] * 10**6
            list_a.append(Sigma(theta,d,f,c)**rx * Omega(tau,f_d)**subc)
    arr_a = np.array(list_a).reshape([-1,1])
    return(arr_a) #np.hstack([arr_a]*30))
def Pmu(mat_en1,theta,tau):
    mat_a = SteerVec(theta,tau)
    return(1 / np.transpose(np.conjugate(mat_a)).dot(mat_en1).dot(np.transpose(np.conjugate(mat_en1))).dot(mat_a))
def SmoothCSI(dt_csi):
    list_rxs = []
    for rx in range(3):
        list_rx1 = []
        for subc in range(15):
            list_rx1.append(dt[subc:subc+15,rx])
        list_rxs.append(np.array(list_rx1))

    list_smcsi = []
    for i in range(2):
        csi_arr1 = np.hstack([list_rxs[i],list_rxs[i+1]])
        list_smcsi.append(csi_arr1)
    return(np.vstack([list_smcsi[0],list_smcsi[1]]))
def NoiseVec(mat_x):
    mat_xx = mat_x.dot(np.transpose(np.conjugate(mat_x)))
    eigv, mat_en = np.linalg.eig(mat_xx)
    mat_en1 = mat_en[np.argmin(eigv),:].reshape([-1,1])
    return(mat_en1)

# read sample data
# load and uncompress.
with gzip.open(path_csi+file,'rb') as f:
    data1 = pickle.load(f)
data1 = data1**4 / (np.abs(data1)**3)
data1 = np.nan_to_num(data1)

csi_tx1 = data1[:,:,0,:]
csi_tx2 = data1[:,:,1,:]

# make music array
dt = csi_tx1[1000]
mat_x = SmoothCSI(dt)
mat_en1 = NoiseVec(mat_x)

# th, tau range

th_arr = np.linspace(0,2*math.pi,360)
ta_arr = np.linspace(0,10,360)

# calc music
mat_music = np.zeros([360,360]).astype('float32')
for i,ta in enumerate(ta_arr):
    for j,th in enumerate(th_arr):
        mat_music[i,j] = 20*math.log10(np.abs(Pmu(mat_en1,th,ta)))

abs_music = mat_music
th_music = np.mean(abs_music,axis=0)
ta_music = np.mean(abs_music,axis=1)


#plt.plot(th_music)
#plt.plot(ta_music)

ax = sns.heatmap(abs_music)#,vmin=np.median(sig_mat))
fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(0,360)
Y = np.arange(0,360)
X,Y = np.meshgrid(X,Y)
Z = abs_music

surf = ax.plot_surface(X,Y,Z,cmap='coolwarm',linewidth=10,antialiased=True)
#wire = ax.plot_wireframe(X,Y,Z,color='r',linewidth=0.1)
fig.colorbar(surf,shrink=0.5,aspect=5)
fig.tight_layout()
plt.show()


