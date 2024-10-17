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
fs    =    5.7E-20 # # pa-3 m2 s-1 # this value and dimension is only correct for n=3
years = 100

# adjusted: elalist and elayear. See source file in main directory
elalist = np.linspace(1400., 2200., 9)  # m
elayear = np.full_like(elalist, years, dtype=int)  # years
beta    =    0.007    # [m/m]/yr
maxb    =    2.      # m/yr



# Initialisation: start from zero ice and define 
def get_bedrock(xaxis):
    # here you put in your own equation that defines the bedrock
    bedrock = 2000. - xaxis*0.08
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
nyear    = int(np.sum(elayear))
nela     = np.size(elalist)
if nela != np.size(elayear):
    print("the arrays of elalist and elayear do not have the same length!")
    exit()
else:
    elaswch = np.zeros(nela)
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
mass = np.arange(nyear+1)
mass_change_ela = []
mass_initial = []

# (re)set initial values so that the accumulation area has glacier right away.
hice = np.where(bedrock>ela, np.where(hice<0.11, 0.11, hice), hice)
lengthmem[0] = np.sum(np.where(hice>0.1, dx, 0.))
volumemem[0] = np.sum(hice)*dx
elamemory[0] = ela

cd    = 2./5.*fd*(rho*g)**3  # adjusted
cs    = fs*(rho*g)**3  # adjusted
#print(ntpy*nyear+1)
#print(FluxAtPoints)
#0----------------------------------------------------------------------------- loop over timesteps
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
        # lists the elements of elaswch that are equal or smaller than it
        [ielanow] = np.nonzero(elaswch<=it) 
        # the last one is the current ela
        ela       = elalist[ielanow[-1]]
        ### mass calculation
        iy = it//ntpy  
        mass_bef = np.sum(hice) * dx * rho
        #mass calculateion at the beginning of each ela 
        if it % (years * ntpy) == 1:  # Every 100 years
            mass_initial.append(mass_bef)
            print(f"Year {iy}: Mass initial = {mass_bef}, ELA = {ela}")

        
    smb[:] = (h-ela)*beta
    smb[:] = np.where(smb>maxb, maxb, smb) 
    
    hice += smb/ntpy
    hice += dt*dFdx # minus sign added
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
        #print(f"Year {iy}: Mass = {mass[iy]}, ELA = {elamemory[iy]}")
        # mass after ELA change
        if it % (years * ntpy) == 0:  # Every 100 years and 200 timesteps
            mass_change_ela.append(mass[iy])
            print(f"Year {iy}: Mass final = {mass[iy]}, ELA = {elamemory[iy]}")

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

#------------------------------------------------------------------------------       

print(np.size(mass_initial))
print(mass_initial)
print(np.size(mass_change_ela))
print(mass_change_ela)
print("Calculating change: ")
mass_change = [a - b for a, b in zip(mass_change_ela, mass_initial)]

print(mass_change)


#-------------------------------------------------------------------------------

# at this point, the simulation is completed.        
# the following is needed to make the animation        
fig  = plt.figure()
ax1  = fig.add_subplot(311, autoscale_on=False, xlim=(0,totL/1000.), \
                      ylim=(np.min(bedrock),np.max(hsurfmem)+10.))
mina2 = min(np.min(smbmem),np.min(ifdmem))
maxa2 = max(np.max(smbmem),np.max(ifdmem))
ax2   = fig.add_subplot(312, autoscale_on=False, xlim=(0,totL/1000.), \
                      ylim=(mina2,maxa2))
mina3 = min(np.min(fldmem),np.min(flsmem))
maxa3 = max(np.max(fldmem),np.max(flsmem))
ax3   = fig.add_subplot(313, autoscale_on=False, xlim=(0,totL/1000.), \
                      ylim=(mina3,maxa3))


# define the line types
bedrline, = ax1.plot([],[],'-', c='saddlebrown') 
hsrfline, = ax1.plot([],[],'-', c='navy')
time_template = 'time = %d y'
time_text = ax1.text(0.5, 0.92, '', transform=ax1.transAxes )
smbline,  = ax2.plot([],[],'-', c='navy')
ifdline,  = ax2.plot([],[],'-', c='red')
fxdline,  = ax3.plot([],[],'-', c='navy')
fxsline,  = ax3.plot([],[],'-', c='red')

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
    

#------------------------------------------------------------------------------ 
# postprocessing - estimating responsetime after knowing the change
def get_responsetime(dataarray, initial_year):  
    print('Define your function to estimate the response time for the change after year {0:4d}'.format(initial_year))
    response_time = 0.     

    return response_time, initial_year + response_time

ResponseTimes = np.zeros(nela)
ResponseYears = np.zeros(nela)

# The first ela value is excluded from the analysis as that has the spin-up
# Here, the length is used for the responsetime. One could also take the mass. 
#  If desired, do not use lengthmem but volumemem.
ys = elayear[0]
for i in range(1,nela):
    if i == nela-1:
        ye = nyear
    else:
        ye = ys + elayear[i]
    ResponseTimes[i], ResponseYears[i] = get_responsetime(lengthmem[ys:ye], ys)
        
    ys = ye



fig2,ax2a = plt.subplots()
#ax2a.plot(yearlist,lengthmem/1000. ,'k')
ax2a.plot(yearlist,volumemem/1000. ,'k')
ax2a.set_xlabel('Model year (yr)')
ax2a.set_ylabel('Glacier volume (km^3)')
ax2a.set_xlim([0, nyear])
lmima = [ np.min(lengthmem/1000.), np.max(lengthmem/1000.) ]
for i in range(1,nela):
    ax2a.plot([ResponseYears[i],ResponseYears[i]], lmima, 'b')

ax2b  = ax2a.twinx()
color = 'tab:red'
ax2b.plot(yearlist, elamemory, color=color)
ax2b.set_ylabel('Ela', color=color)
ax2b.tick_params(axis='y', labelcolor=color)   
fig2.tight_layout()
fig2.savefig('..\figures\glacierlength.png', dpi=300)
    
plt.show()

fig3,ax3 = plt.subplots()
ax3.scatter(elalist[1:], ResponseTimes[1:])
fig3.savefig('../figures/responsetime.png', dpi=300)        

   

                        

                        
