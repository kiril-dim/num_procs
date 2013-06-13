#implementation of Crank-Nicolson
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import array
import numpy as np
from tridiag_solver_c import SolveTridiagonal
import time

# Solving du/dt  = A x d^2/dx^2
A	   = 0.04

t_step = 0.01

x_step = 0.05
x_begin 	= -2
x_end		= 2
x_steps = int((x_end - x_begin)/x_step)

##################### Calculate analytic solution ######################
# Reference http://en.wikipedia.org/wiki/Heat_equation under
# Fundamental solutions

def phi(x,t):
	if t == 0:
		return np.zeros_like(x)
	return np.exp(-(x*x)/(4*A*t))/np.sqrt(4*np.pi*A*t)

# Analytical solution for the given equation
# NB: Needs verification
def get_sol(t):
	x = np.linspace(x_begin,x_end,x_steps)
	Y = x
	ut = np.zeros(x_steps)
	for i in xrange(x_steps):
		 X = np.ones(x_steps)*x[i]
		 ut[i] = (phi(X - Y,t)*initial_values*4).mean()
	return ut
		
########################################################################


################## Finite-Difference solution ##########################
#Solve heat equation for its simplest form
# du/dt  = A x d^2/dx^2. This gives rise to the
# following finite-difference scheme
# Set r = (A t_step) / (2 (x_step)^2 )
# Then - r U[n+1][i+1] + (1+2*r)*U[n+1][i] - r * U[n+1][i-1] =
#	= r U[n][i+1] + (1-2*r)*U[n][i] + r * U[n][i-1]



# Beginning of time is assument to be 0
T_end  		= 50


# weighing coefficient
lmbd = 0.0

n_steps = int(T_end/t_step)

# Define initial values
initial_values = np.zeros(x_steps)
initial_values[x_steps/2] = 5

print "Number of time steps: ",n_steps
print "Number of x points: ",x_steps

# Set up the grid
U = np.zeros((n_steps,x_steps))
U[0] = initial_values


r = (A*t_step) / (2*x_step**2)

# Define our triadiagonal matrix
# sub diagonal is the same as super diagonal

subDiag = np.ones(x_steps) * (-r)
subDiag[0] = 0
supDiag = np.ones(x_steps) * (-r) 
supDiag[-1] = 0
Diag = np.ones(x_steps) * (1 + 2*r) 

# Function to calculate Y for the above imputs
def calculateY(arr):
	assert len(arr) > 2
	ret = np.zeros(len(arr))
	ret[1:-1] = r * arr[0:-2] + (1-2*r) * arr[1:-1] + r * arr[2::]
	ret[0]    = r * arr[0]    + (1-2*r) * arr[1]
	ret[-1]   = r * arr[-1]   + (1-2*r) * arr[-2]
	return ret

# Do the stepping
t1 = time.time()

x = np.linspace(x_begin,x_end,x_steps)
for i in xrange(n_steps-1):	
	y = calculateY(U[i].copy())
	U[i+1] = SolveTridiagonal(subDiag,Diag,supDiag,y)
	
t2 = time.time()

print "Loop finished in ",t2-t1


# First set up the figure, the axis, and the plot element we want to animate
# line1 is the finite-difference solution and line2 is the analytic solution
# when it works they should be indistiguishable.
# We calculate the analytic solution directly whilst generating the graph

fig = plt.figure()
ax = plt.axes(xlim=(-4.1, 4.1), ylim=(0, 1.3))
line, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    line2.set_data([],[])
    return line, line2,

# animation function.  This is called sequentially
def animate(i):
    line.set_data(x, U[i])
    line2.set_data(x, get_sol(i*t_step))
    return line, line2

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=n_steps, interval=3, blit=True)

plt.show()
