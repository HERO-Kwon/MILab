import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg-4.1-win64-static\\bin\\ffmpeg.exe'

def plot_animated(exp_id,arr_scaled):

    #Draw Graph
    fig = plt.figure()
    ax00 = fig.add_subplot(221)
    ax01 = fig.add_subplot(222)
    ax10 = fig.add_subplot(223)
    ax11 = fig.add_subplot(224)

    fig.suptitle('Animated plot of : '+exp_id)
    ax00.set_title('Antenna1')
    ax01.set_title('Antenna2')
    ax00.set_ylabel('Abs')
    ax10.set_ylabel('Phase')
    ax10.set_xlabel('Subcarrier')
    ax11.set_xlabel('Subcarrier')

    ax00.set_ylim(0,50)
    ax01.set_ylim(0,50)

    x = np.arange(0,30)
    line000, = ax00.plot(x,np.abs(arr_scaled[1,:,0,0]))
    line001, = ax00.plot(x,np.abs(arr_scaled[1,:,0,1]))
    line002, = ax00.plot(x,np.abs(arr_scaled[1,:,0,2]))
    line010, = ax01.plot(x,np.abs(arr_scaled[1,:,1,0]))
    line011, = ax01.plot(x,np.abs(arr_scaled[1,:,1,1]))
    line012, = ax01.plot(x,np.abs(arr_scaled[1,:,1,2]))
    line100, = ax10.plot(x,np.angle(arr_scaled[1,:,0,0]))
    line101, = ax10.plot(x,np.angle(arr_scaled[1,:,0,1]))
    line102, = ax10.plot(x,np.angle(arr_scaled[1,:,0,2]))
    line110, = ax11.plot(x,np.angle(arr_scaled[1,:,1,0]))
    line111, = ax11.plot(x,np.angle(arr_scaled[1,:,1,1]))
    line112, = ax11.plot(x,np.angle(arr_scaled[1,:,1,2]))

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
        line000.set_ydata(np.abs(arr_scaled[i,:,0,0]))  # update the data.
        line001.set_ydata(np.abs(arr_scaled[i,:,0,1]))
        line002.set_ydata(np.abs(arr_scaled[i,:,0,2]))
        line010.set_ydata(np.abs(arr_scaled[i,:,1,0]))  # update the data.
        line011.set_ydata(np.abs(arr_scaled[i,:,1,1]))
        line012.set_ydata(np.abs(arr_scaled[i,:,1,2]))    
        line100.set_ydata(np.angle(arr_scaled[i,:,0,0]))  # update the data.
        line101.set_ydata(np.angle(arr_scaled[i,:,0,1]))
        line102.set_ydata(np.angle(arr_scaled[i,:,0,2]))
        line110.set_ydata(np.angle(arr_scaled[i,:,1,0]))  # update the data.
        line111.set_ydata(np.angle(arr_scaled[i,:,1,1]))
        line112.set_ydata(np.angle(arr_scaled[i,:,1,2]))    
        return [line000,line001,line002,line010,line011,line012,line100,line101,line102,line110,line111,line112],


    ani = animation.FuncAnimation(
        fig, animate, init_func=init,interval=1,blit=False, save_count=100,repeat=False)

    FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
    ani.save(exp_id+'_ani.mp4', writer = FFwriter)

    return fig