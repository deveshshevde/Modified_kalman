import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np



#https://www.youtube.com/watch?v=jn8vQSEGmuM
#https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume11/fox99a-html/node3.html#SECTION00021000000000000000
#https://ros-developer.com/2017/12/11/markov-localization-explained/


time_delay = 0.1 #can change but carefully
x_gy = []#will come from devanshu bhaiya old code
y_gy = []
z_gy = []
x_kal = []
y_kal = []
z_kal = []


#creted plot for plootinf gaussian plot//
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

var_in_x = 0.01
var_in_y = 0.01
var_in_z = 0.01
var_during_filtering_x = 0.01
var_during_filtering_y = 0.01
var_during_filtering_z = 0.01

#https://www.bauer.uh.edu/rsusmel/phd/ec2-8.pdf // can recheck from here



X = np.matrix([[0], [0], [9.81]]) #//assuming 9.81 m/s^2 acc. in Z
P = np.matrix([
               [var_during_filtering_x**2, 0, 0],
               [0, var_during_filtering_y**2, 0],
               [0, 0, var_during_filtering_z**2],
              ])
A = np.identity(3)
A_Trans = np.transpose(A)
B = np.identity(3)
u = np.matrix([[0], [0], [0]])
w = np.matrix([[0], [0], [0]])
Q = np.matrix([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
H = np.identity(3)
H_T = np.transpose(H)
R = np.matrix([
               [var_in_x**2, 0, 0],
               [0, var_in_y**2, 0],
               [0, 0, var_in_z**2],
              ])
I = np.identity(3)


def kalman(Y):#acc data as argu //
    # predict state and process matrix for current iteration
    X_predict = (A * X) + (B * u)+ w
    P_predict = (A * P * A_Trans) - Q# Q -> noise added to kalmaan //

    # find kalman gain
    K = P_predict * H_T * (np.linalg.inv(H * P_predict * H_T + R))

    # estimate new state and process matricies
    X_estimate = X_predict + K * (Y - H * X_predict)
    P_estimate = (I - K * H) * P
    return X_estimate, P_estimate


def animate(i, t, x_measured, y_measured, z_measured, x_kalman, y_kalman, z_kalman):


    # collects accelerometer data

    accel_x = []#data must be added here for gaussian 1
    accel_y = []#data must be added here for gaussian 1
    accel_z =[]#data must be added here for gaussian 1

    # processes accelerometer data to kalman filter
    Y = np.matrix([[accel_x], [accel_y], [accel_z]])
    X, P = kalman(Y)
    kal_x = X[0, 0]
    kal_y = X[1, 0]
    kal_z = X[2, 0]

  #data coming from gaussian
    x_measured.append(accel_x)
    y_measured.append(accel_y)
    z_measured.append(accel_z)
    x_kalman.append(kal_x)
    y_kalman.append(kal_y)
    z_kalman.append(kal_z)
    t.append(i * time_delay)

   #bayesian data //
    x_measured = x_measured[-20:]#dont make it too low data accuracy will change
    y_measured = y_measured[-20:]
    z_measured = z_measured[-20:]
    x_kalman = x_kalman[-20:]
    y_kalman = y_kalman[-20:]
    z_kalman = z_kalman[-20:]
    t = t[-20:]

    # graphs data

    #product of bayesian
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    ax.clear()
    ax.plot(t, x_measured, color='g', label='measured X')
    ax.plot(t, x_kalman, linestyle='dashed', color='g', label='kalman X')
    ax.plot(t, y_measured, color='r', label='measured Y')
    ax.plot(t, y_kalman, linestyle='dashed', color='r', label='kalman Y')
    ax.plot(t, z_measured, color='b', label='measured Z')
    ax.plot(t, z_kalman, linestyle='dashed', color='b', label='kalman Z')

#dont know how to plot // google se chapa
try:
    ani = animation.FuncAnimation(fig, animate,
                                  fargs=(x_gy, y_gy, z_gy, x_kal, y_kal, z_kal),
                                  interval=int(time_delay * 1000))
    plt.show()
except KeyboardInterrupt:
    print("\nEnd of Kalman Filter for MPU6050!")
    plt.close()
