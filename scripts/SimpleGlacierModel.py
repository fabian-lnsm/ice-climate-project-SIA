# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


totL  = 20000 # total length of the domain [m]
dx    =   100 # grid size [m]

ntpy  =   200 # number of timesteps per year

ZeroFluxBoundary = False # "Your choice" # either no-flux (True) or No-ice boundary (False)
FluxAtPoints     = False #"Your choice" # if true, the ice flux is calculated on grid points, 
                        #  otherwise on half points
StopWhenOutOfDomain = True                        
                        
ndyfigure = 5           # number of years between a figure frame                        


rho   =  917.      # kg/m3
g     =    9.80665 # m/s2
fd    =    1.9E-24 # # pa-3 s-1 # this value and dimension is only correct for n=3
fs    =    5.7E-20 # # pa-3 m2 s-1 # this value and dimensmepawoion is only correct for n=3
years = 500

# adjusted: elalist and elayear. See source file in main directory
elalist = np.linspace(1400., 1600., 9)  # m
elayear = np.full_like(elalist, years, dtype=int)  # years
beta    =    0.007    # [m/m]/yr
maxb    =    2.      # m/yr



# Initialisation: start from zero ice and define 
def get_bedrock(xaxis):
    # here you put in your own equation that defines the bedrock
    bedrock = 1700. -xaxis*0.05 + 100.* np.exp(-xaxis*0.001)
    #bedrock = 1700. -xaxis*0.05 + 300.* np.exp(-xaxis*0.001)
    #bedrock = 1800. - xaxis*0.05
    return bedrock

# Start calculations
# constants that rely on input
nx    = int(totL/dx)
dx    = totL/nx       # redefine, as it     
xaxis = np.linspace(0,totL,nx,False) + dx*0.5
xhaxs = np.linspace(dx,totL,nx-1,False) / 1000.

bedrock = get_bedrock(xaxis)

dt    = 365.*86400./ntpy # length of one timestep in seconds!

hice   = np.zeros(nx)    # ice thickness
dhdx   = np.zeros(nx)    # the local gradient of h
fluxd  = np.zeros(nx+2)  # this will be the flux per second!!!!
fluxs  = np.zeros(nx+2)  # this will be the flux per second!!!!
dFdx = np.zeros(nx)    # change in ice thickness due to the ice flux, per second
smb    = np.zeros(nx)

# preparations for the ela-selection
# elaswch is a list of time steps on which a new ela value should be used.
nyear    = int(np.sum(elayear)) + 500
nela     = np.size(elalist)
if nela != np.size(elayear):
    print("the arrays of elalist and elayear do not have the same length!")
    exit()
else:
    elaswch = np.zeros(nela)
    elaswch[0]=500*ntpy
    for i in range(0,nela-1):
        elaswch[i+1] = elaswch[i] + (elayear[i]*ntpy)
    ela     = elalist[0]
        
# preparations for the animation
nframes  = nyear//ndyfigure + 1
hsurfmem = np.zeros([nx,nframes])
smbmem   = np.zeros([nx,nframes])
ifdmem   = np.zeros([nx,nframes])
fldmem   = np.zeros([nx-1,nframes])
flsmem   = np.zeros([nx-1,nframes])
iframes  = 0

# preparations for response time calculations
lengthmem = np.zeros(nyear+1)
volumemem = np.zeros(nyear+1)
elamemory = np.zeros(nyear+1)
yearlist  = np.arange(nyear+1)

#mass calculation
mass = np.arange(nyear+1, dtype=float)
mass_change_ela = []
mass_initial = []

# (re)set initial values so that the accumulation area has glacier right away.
hice = np.where(bedrock>ela, np.where(hice<0.11, 0.11, hice), hice)
lengthmem[0] = np.sum(np.where(hice>0.1, dx, 0.))
volumemem[0] = np.sum(hice)*dx
elamemory[0] = ela
highest_point_x=[]
highest_point_h=[]
end_flow_line_x=[]
end_flow_line_h=[]
ela_list = []

cd    = 2./5.*fd*(rho*g)**3  # adjusted
cs    = fs*(rho*g)**3  # adjusted

def calculate_flow_line(hsurfmem, bedrock, dx, iframes): # evaluate highest and smalles value and calculate slope through these points   
 
    hsurface = hsurfmem[:, iframes]  # Glacier surface at the final time frame
    highest_point_index = np.argmax(hsurface)  # Index of the highest surface point
    
    # initial position
    initial_x = highest_point_index * dx / 1000.0
    initial_h = hsurface[highest_point_index]

###
    surface_minus_bedrock = hsurface - bedrock
    right_difference = surface_minus_bedrock[highest_point_index + 1:]  # Values to the right of the highest point because to the left there is always a smaller one 

    # Check if there are values to the right
    if right_difference.size > 0:
        index_lowest = np.argmin(right_difference) + (highest_point_index + 1)  # Adjust index to the original array
        
        end_x = index_lowest * dx / 1000.0
        end_h = hsurface[index_lowest]
    
    return initial_x, initial_h, end_x, end_h


#----------------------------------------------------------------------------- loop over timesteps
print("Run model for {0:3d} years".format(nyear))
for it in range(1, ntpy*nyear+1):
    h = hice + bedrock
    if FluxAtPoints:
        '''
        dhdx[1:-1] = (h[2:]-h[:-2])/(2*dx)
        
        # the following equations needs to be adjusted according to your discretisation
        # note that flux[1] is at the point 0
        fluxd[1:-1] = cd * (dhdx) * (hice)  
        fluxs[1:-1] = cs * (dhdx) * (hice)

        # derive flux convergence
        dFdx[:]  = (fluxd[2:]-fluxd[:-2]+fluxs[2:]-fluxs[:-2])/(2*dx)
        '''
    else:
        # the following equations needs to be adjusted according to your discretisation
        dhdx[:-1]  = ((h[1:]-h[:-1])/dx) # so 0 is at 1/2 actually
        # note that flux[1] is at the point 1/2
        fluxd[1:-2] = cd * (dhdx[:-1])**3 * (0.5*hice[1:] + 0.5 * hice[:-1])**5
        fluxs[1:-2] = cs * (dhdx[:-1])**3 * (0.5*hice[1:] + 0.5 * hice[:-1])**3
        
        # derive flux convergence
        dFdx[:]  = (fluxd[1:-1]-fluxd[:-2] + fluxs[1:-1]-fluxs[:-2])/dx
        
    # calculate smb (per year)
    # first update ela (once a year) 
    if it%ntpy == 1:
        if it>500*ntpy:
            # lists the elements of elaswch that are equal or smaller than it
            [ielanow] = np.nonzero(elaswch<=it) 
            # the last one is the current ela
            ela       = elalist[ielanow[-1]]
            ### mass calculation
        iy = it//ntpy
        mass_bef = np.sum(hice) * dx * rho

        #mass calculation at the beginning of each ela 
        if it % (years * ntpy) == 1:  # Every 100 years
            mass_initial.append(mass_bef)
            print(f"Year {iy}: Mass initial = {mass_bef}, ELA = {ela}")
            flow_line_x, flow_line_h, flow_line_end_x, flow_line_end_h = calculate_flow_line(hsurfmem, bedrock, dx, iframes)
            #slope_val, offset_val = calculate_flow_line(hsurfmem, bedrock, dx, iframes)
            highest_point_x.append(flow_line_x)
            highest_point_h.append(flow_line_h)
            end_flow_line_x.append(flow_line_end_x)
            end_flow_line_h.append(flow_line_end_h)
            ela_list.append(ela)


    smb[:] = (h-ela)*beta
    smb[:] = np.where(smb>maxb, maxb, smb) 
    
    hice += smb/ntpy
    hice += dt*dFdx 
    hice[:] = np.where(hice<0., 0., hice) # remove negative ice thicknesses
    if ZeroFluxBoundary == False:
        hice[0] = hice[-1] = 0.
    
    if it%ntpy == 0:
        if np.any(np.isnan(hice)):
            print('Values got NaN!')
            break
        iy = it//ntpy
        lengthmem[iy] = np.sum(np.where(hice>0.1, dx, 0.))
        volumemem[iy] = np.sum(hice)*dx
        elamemory[iy] = ela
        mass[iy] = volumemem[iy] * rho
        # mass after ELA change
        if it % (years * ntpy) == 0:  # Every 100 years and 200 timesteps
            mass_change_ela.append(mass[iy])
            print(f"Year {iy}: Mass final = {mass[iy]}, ELA = {elamemory[iy]}")
            #print(f'Save mass change: {mass_change_ela}')

    if it%(ndyfigure*ntpy) == 0:
        iframes            += 1
        hsurfmem[:,iframes] = hice + bedrock
        smbmem[:,iframes]   = smb
        ifdmem[:,iframes]   = dFdx[:]*365.*86400.
        fldmem[:,iframes]   = -fluxd[1:-2]*365.*86400.
        flsmem[:,iframes]   = -fluxs[1:-2]*365.*86400.
        if StopWhenOutOfDomain:
            if hice[-1]>1.:
                print("Ice at end of domain!")
                break

mass_change_ela = np.array(mass_change_ela[1:]) #leave out the mass after 500 years (no ela-switch here)
#mass_change = [a - b for a, b in zip(mass_change_ela, mass_initial)]
mass_change = mass_change_ela[1:] - mass_change_ela[:-1]
#print(mass_change)
#print(mass_initial)

#--------------------------------- Animation -----------------------------

# at this point, the simulation is completed.        
# the following is needed to make the animation        
fig  = plt.figure()
ax1  = fig.add_subplot(311, autoscale_on=False, xlim=(0,totL/1000.), \
                      ylim=(np.min(bedrock),np.max(hsurfmem)+10.))
mina2 = min(np.min(smbmem),np.min(ifdmem))
maxa2 = max(np.max(smbmem),np.max(ifdmem))
ax2   = fig.add_subplot(312, autoscale_on=False, xlim=(0,totL/1000.), \
                      ylim=(mina2,maxa2), sharex=ax1)
mina3 = min(np.min(fldmem),np.min(flsmem))
maxa3 = max(np.max(fldmem),np.max(flsmem))
ax3   = fig.add_subplot(313, autoscale_on=False, xlim=(0,totL/1000.), \
                      ylim=(mina3,maxa3), sharex=ax1)
ax1.set_ylabel('Elevation [m]')
ax2.set_ylabel('[m/yr]')
ax3.set_ylabel(r'Flux [$m^2$/yr]')
ax3.set_xlabel('Length [km]')
ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)



# define the line types
bedrline, = ax1.plot([],[],'-', c='saddlebrown', label = 'Bedrock') 
hsrfline, = ax1.plot([],[],'-', c='navy', label = 'Glacier surface')
ax1.legend()
time_template = 'time = %d y'
time_text = ax1.text(0.5, 0.92, '', transform=ax1.transAxes )
smbline,  = ax2.plot([],[],'-', c='navy', label = 'SMB')
ifdline,  = ax2.plot([],[],'-', c='red', label = 'dF/dx')
ax2.legend()
fxdline,  = ax3.plot([],[],'-', c='navy', label = 'Deformation flux')
fxsline,  = ax3.plot([],[],'-', c='red', label = 'Sliding flux')
ax3.legend()

# initialize the animation
def init_anim():
    bedrline.set_data([], [])
    hsrfline.set_data([], [])
    time_text.set_text('')
    smbline.set_data([], [])
    ifdline.set_data([], [])
    fxdline.set_data([], [])
    fxsline.set_data([], [])
    return bedrline, hsrfline, time_text, smbline, ifdline, fxdline, fxsline

# update the animation with data for saved frame #tf
def animate(tf):
    bedrline.set_data(xaxis/1000., bedrock)
    hsrfline.set_data(xaxis/1000., hsurfmem[:,tf])
    time_text.set_text(time_template % int(tf*ndyfigure))
    smbline.set_data(xaxis/1000.,  smbmem[:,tf])
    ifdline.set_data(xaxis/1000.,  ifdmem[:,tf])
    fxdline.set_data(xhaxs      ,  fldmem[:,tf])
    fxsline.set_data(xhaxs      ,  flsmem[:,tf])
    return bedrline, hsrfline, time_text, smbline, ifdline, fxdline, fxsline
    
# call and run animation
ani = animation.FuncAnimation(fig, animate, np.arange(iframes),\
         interval=25, blit=True, init_func=init_anim, )     

# SAVING PYTHON MOVIES IS PLATFORM DEPENDEND.     
writervideo = animation.FFMpegWriter(bitrate=2000, fps = 45)
ani.save('../movies/animation.mp4', writer=writervideo, dpi=400)
print('Movie has been saved')
#-------------------------- Response Time -----------------------------------
def get_responsetime(dataarray, initial_year):  
    delta_length = dataarray[0] - dataarray[-1]
    target = delta_length * (1-1/np.e)
    response_time = np.argmax((dataarray[0]-dataarray) > target)   

    return response_time, initial_year + response_time

ResponseTimes = np.zeros(nela)
ResponseYears = np.zeros(nela)

# The first ela value is excluded from the analysis as that has the spin-up
# Here, the length is used for the responsetime. One could also take the mass. 
#  If desired, do not use lengthmem but volumemem.
ys = 500
for i in range(0,nela):
    ye = ys + elayear[i]
    ResponseTimes[i], ResponseYears[i] = get_responsetime(lengthmem[ys:ye], ys)
    #print(ys, ye,ResponseTimes)    
    ys = ye

#-------------------------------- Volume Vs Time -------------------------------

fig2,ax2a = plt.subplots()
ax2a.grid(True, alpha=0.5)
ax2a.plot(yearlist,volumemem,'k')
ax2a.set_xlabel('Model years')
ax2a.set_ylabel(r'Glacier volume [$km^3$]')
ax2a.set_xlim([0, nyear])
lmima = [ np.min(lengthmem/1000.), np.max(lengthmem/1000.) ]
for i in range(1,nela):
    ax2a.plot([ResponseYears[i],ResponseYears[i]], lmima, 'b')

ax2b  = ax2a.twinx()
color = 'tab:red'
ax2b.plot(yearlist, elamemory, color=color)
ax2b.set_ylabel('ELA [m]', color=color)
ax2b.tick_params(axis='y', labelcolor=color)   
fig2.tight_layout()
fig2.savefig('../figures/glacierlength.png', dpi=300)

#-------------------------------- Response Time Plot --------------------------       

fig3,ax3 = plt.subplots()
ax3.grid(True, alpha=0.3)
ax3.scatter(elalist[1:], ResponseTimes[1:], color='purple', label='Observed response times', s=35)
ax3.hlines(np.mean(ResponseTimes[1:]), elalist[1], elalist[-1], color='black', label='Mean response time', linestyles='dashed')
ax3.set_xlabel('ELA [m]')
ax3.set_ylabel('Time [yr]')  
ax3.set_ylim([100, 200])
ax3.legend()
fig3.savefig('../figures/responsetimes.png', dpi=300)    

#-------------------------------- Mass Change ---------------------------------      

fig4,ax4 = plt.subplots()
print('ELA: ',elalist[1:])
print('Mass change: ', mass_change[:])
ax4.scatter(elalist[1:], mass_change[:])
ax4.set_xlabel('ELA [m]')
ax4.set_ylabel('Glacier mass [kg]')
ax4.grid(True, alpha=0.5)
fig4.savefig('../figures/masschanges.png', dpi=300)


#-------------------------------- Flow Line ------------------------------------      
#Plot the flow line on top of the glacier surface
fig5, ax5 = plt.subplots()
for i in range(3, len(highest_point_x)):
    ax5.plot([highest_point_x[i], end_flow_line_x[i]], [highest_point_h[i], end_flow_line_h[i]], label=f'ELA = {ela_list[i]:.0f}', linestyle='-', marker=None)
ax5.set_xlabel('Distance [km]')
ax5.set_ylabel('Elevation [m]')
ax5.grid(True, alpha=0.5)
ax5.legend()
fig5.savefig('../figures/flowlines.png', dpi=300)

