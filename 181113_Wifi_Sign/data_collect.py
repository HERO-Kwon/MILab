import scipy.io as sio
import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg-4.1-win64-static\\bin\\ffmpeg.exe'
%matplotlib notebook

folder_path = 'D:\Data\Wi_Fi Dataset\Wi-Fi dataset_CSI\\'
folder_list = os.listdir(folder_path)

df_subc = pd.DataFrame()
for exp_id in folder_list:
    file_path = folder_path + exp_id
    file_list = os.listdir(file_path)

    df_sc = pd.DataFrame()
    dict_csi = {}
    scalar_colname = ['timestamp_low','bfee_count','Nrx','Ntx','rssi_a','rssi_b','rssi_c','noise','agc','rate']
    for i, file in enumerate(file_list):
        data_read = sio.loadmat(os.path.join(file_path,file))
        num_search = re.search('\d+',file)
        csi_num = int(num_search.group(0))

        # scalar data
        read_sc0_8 = [data_read['csi_entry'][0][0][a][0][0] for a in range(9)]
        data_sc = pd.DataFrame(read_sc0_8).astype('int').T
        data_sc.columns=scalar_colname[0:9]
        data_sc['perm'] = pd.Series([data_read['csi_entry'][0][0][9][0]])
        data_sc['rate'] = data_read['csi_entry'][0][0][10][0][0]
        data_sc.index = [csi_num]

        # csi info
        csi_raw = data_read['csi_entry'][0][0][11]
        csi_scaled = data_read['csi_entry'][0][0][12]

        # aggregate
        df_sc = df_sc.append(data_sc)
        dict_csi[csi_num] = csi_raw,csi_scaled
    
    #Draw Graph
    fig = plt.figure()
    ax00 = fig.add_subplot(221)
    ax01 = fig.add_subplot(222)
    ax10 = fig.add_subplot(223)
    ax11 = fig.add_subplot(224)

    fig.suptitle('Animated plot of : '+exp_id)
    ax00.set_title('Receiver1')
    ax01.set_title('Receiver2')
    ax00.set_ylabel('Abs')
    ax10.set_ylabel('Phase')
    ax10.set_xlabel('Subcarrier')
    ax11.set_xlabel('Subcarrier')

    ax00.set_ylim(0,50)
    ax01.set_ylim(0,50)

    x = np.arange(0,30)
    line000, = ax00.plot(x,np.abs(dict_csi[1][1][0][0]))
    line001, = ax00.plot(x,np.abs(dict_csi[1][1][0][1]))
    line002, = ax00.plot(x,np.abs(dict_csi[1][1][0][2]))
    line010, = ax01.plot(x,np.abs(dict_csi[1][1][1][0]))
    line011, = ax01.plot(x,np.abs(dict_csi[1][1][1][1]))
    line012, = ax01.plot(x,np.abs(dict_csi[1][1][1][2]))
    line100, = ax10.plot(x,np.angle(dict_csi[1][1][0][0]))
    line101, = ax10.plot(x,np.angle(dict_csi[1][1][0][1]))
    line102, = ax10.plot(x,np.angle(dict_csi[1][1][0][2]))
    line110, = ax11.plot(x,np.angle(dict_csi[1][1][1][0]))
    line111, = ax11.plot(x,np.angle(dict_csi[1][1][1][1]))
    line112, = ax11.plot(x,np.angle(dict_csi[1][1][1][2]))

    def init():  # only required for blitting to give a clean slate.
        line000.set_ydata([np.nan] * len(x))
        line001.set_ydata([np.nan] * len(x))
        line002.set_ydata([np.nan] * len(x))
        line010.set_ydata([np.nan] * len(x))
        line011.set_ydata([np.nan] * len(x))
        line012.set_ydata([np.nan] * len(x))
        line100.set_ydata([np.nan] * len(x))
        line101.set_ydata([np.nan] * len(x))
        line102.set_ydata([np.nan] * len(x))
        line110.set_ydata([np.nan] * len(x))
        line111.set_ydata([np.nan] * len(x))
        line112.set_ydata([np.nan] * len(x))
        return [line000,line001,line002,line010,line011,line012,line100,line101,line102,line110,line111,line112],

    def animate(i):
        j = i+1
        line000.set_ydata(np.abs(dict_csi[j][1][0][0]))  # update the data.
        line001.set_ydata(np.abs(dict_csi[j][1][0][1]))
        line002.set_ydata(np.abs(dict_csi[j][1][0][2]))
        line010.set_ydata(np.abs(dict_csi[j][1][1][0]))  # update the data.
        line011.set_ydata(np.abs(dict_csi[j][1][1][1]))
        line012.set_ydata(np.abs(dict_csi[j][1][1][2]))    
        line100.set_ydata(np.angle(dict_csi[j][1][0][0]))  # update the data.
        line101.set_ydata(np.angle(dict_csi[j][1][0][1]))
        line102.set_ydata(np.angle(dict_csi[j][1][0][2]))
        line110.set_ydata(np.angle(dict_csi[j][1][1][0]))  # update the data.
        line111.set_ydata(np.angle(dict_csi[j][1][1][1]))
        line112.set_ydata(np.angle(dict_csi[j][1][1][2]))    
        return [line000,line001,line002,line010,line011,line012,line100,line101,line102,line110,line111,line112],


    ani = animation.FuncAnimation(
        fig, animate, init_func=init,interval=1,blit=True, save_count=100,repeat=False)

    FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
    ani.save(exp_id+'_ani.mp4', writer = FFwriter)

    # calc mean, var
    ser_subc = pd.Series()
    sum_subc = np.zeros(dict_csi[1][1].shape)
    sum_sq_subc = np.zeros(dict_csi[1][1].shape)
    len_subc = len(dict_csi)
    for i in dict_csi.keys():
        sum_subc += np.abs(dict_csi[i][1])
        sum_sq_subc += np.abs(dict_csi[i][1])**2
    mean_subc = np.mean(sum_subc/len_subc)
    sd_subc = np.mean(np.sqrt(sum_sq_subc/len_subc - (sum_subc/len_subc)**2))

    ser_subc['exp_id'] = exp_id
    ser_subc['len'] = len_subc
    ser_subc['mean'] = mean_subc
    ser_subc['sd'] = sd_subc

    df_subc = df_subc.append(ser_subc,ignore_index=True)

df_subc.to_csv('df_subc.csv')

