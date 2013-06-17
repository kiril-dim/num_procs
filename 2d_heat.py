# Aim: solve 2D heat equation in at least two different ways:
# 1) Easy way - use directly an optmizer from scipy to generate each frame
# 2) Use finited difference for 2 dimensions - discuss appropriate methods.
 
#implementation of Crank-Nicolson
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib
from numpy import array
import numpy as np
from tridiag_solver_c import SolveTridiagonal
import time

# Solving du/dt  = A x d^2/dx^2 + B x d^2/dy^2
A      = 0.04
B      = 0.04

t_step = 0.1

x_step = 0.1
y_step = 0.1
x_begin     = -1
x_end       = 1
y_begin     = -1
y_end       = 1

x_steps = int((x_end - x_begin)/x_step)
y_steps = int((y_end - y_begin)/y_step)

################## Finite-Difference solution ##########################
#Solve heat equation for its simplest form
# du/dt  = A x d^2/dx^2. This gives rise to the
# following finite-difference scheme
# Alternate direction implicit (ADI) method to two dimensional diffusion equations



# Beginning of time is assument to be 0
T_end       = 5

n_steps = int(T_end/t_step)

# Define initial values
initial_values = np.zeros((x_steps,y_steps))
initial_values[x_steps/2][y_steps/2] = 5

print "Number of time steps: ",n_steps
print "Number of x points: ",x_steps
print "Number of y points: ",y_steps

# Set up the grid
# U = np.zeros((x_steps,y_steps))
U = initial_values

r1 = (A*(t_step/2)) / (2*x_step**2)
r2 = (B*(t_step/2)) / (2*y_step**2)

# Define our triadiagonal matrix
# sub diagonal is the same as super diagonal

subDiag_x = np.ones(x_steps) * (-r1)
subDiag_x[0] = 0
supDiag_x = np.ones(x_steps) * (-r1) 
supDiag_x[-1] = 0
Diag_x = np.ones(x_steps) * (1 + 2*r1) 

subDiag_y = np.ones(y_steps) * (-r2)
subDiag_y[0] = 0
supDiag_y = np.ones(y_steps) * (-r2) 
supDiag_y[-1] = 0
Diag_y = np.ones(y_steps) * (1 + 2*r2)

# Function to calculate Y for the above imputs
def calculateY(arr,r):
    assert len(arr) > 2
    ret = np.zeros(len(arr))
    ret[1:-1] = r * arr[0:-2] + (1-2*r) * arr[1:-1] + r * arr[2::]
    ret[0]    = r * arr[0]    + (1-2*r) * arr[1]
    ret[-1]   = r * arr[-1]   + (1-2*r) * arr[-2]
    return ret

def DoStep(U):
    ret = np.zeros(np.shape(U))
    
    for i in xrange(x_steps):   
        y = calculateY(U[i].copy(),r1)
        U[i] = SolveTridiagonal(subDiag_y,Diag_y,supDiag_y,y)
    
    for j in xrange(y_steps):   
        y = calculateY(U[:,j].copy(),r2)
        ret[:,j] = SolveTridiagonal(subDiag_x,Diag_x,supDiag_x,y)
        
    return ret
    


X = np.arange(x_begin, x_end, x_step)
Y = np.arange(y_begin, y_end, y_step)
X, Y = np.meshgrid(X, Y)


m_R = np.sqrt(X**2 + Y**2)


class plot3dClass( object ):
    '''
    Ploting for 3d surfaces with some dummy data for now.
    '''
    
    def __init__( self, U ):
        self.U = U
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111, projection='3d' )
        self.ax.set_zlim3d( -10e-9, 10e9 )

        self.ax.w_zaxis.set_major_locator( LinearLocator( 10 ) )
        self.ax.w_zaxis.set_major_formatter( FormatStrFormatter( '%.03f' ) )

        self.surf = self.ax.plot_surface(X, Y, U, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        # plt.draw() maybe you want to see this frame?

    def drawNow( self ):
        self.surf.remove()
        self.U = DoStep(self.U)
        self.surf = self.ax.plot_surface(X, Y, self.U, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        plt.draw()                      # redraw the canvas
        time.sleep(1)

matplotlib.interactive(True)

p = plot3dClass(U)
for i in range(10):
    p.drawNow()
