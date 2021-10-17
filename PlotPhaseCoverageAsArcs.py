# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 20:15:04 2021

@author: ar-h1
"""


# def PlotCircleArc(Theta1,Theta2,radius):
    
    
#     theta = np.linspace(Theta1, Theta2, 1000)
    
#     r = np.sqrt(radius)
    
#     x1 = r*np.cos(theta)
#     x2 = r*np.sin(theta)
    
#     plt.plot(x1, x2)
    
#     plt.set_aspect(1)

def GetXYForCircleArc(Theta1,Theta2,radius):
    
    
    theta = np.linspace(Theta1, Theta2, 1000)-np.pi/2
    
    r = radius#np.sqrt(radius)
    
    x1 = r*np.cos(theta)
    x2 = r*np.sin(theta)
    
    return x1,x2
    
    
    



import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

plt.rc('font', family='serif')

# PlotCircleArc(0,2*np.pi,1)

# plt.set_aspect(1)


#theta = np.linspace(0, 2*np.pi, 100)
# theta = np.linspace(np.pi/2, np.pi, 1000)

# r = np.sqrt(1.0)

# x1 = r*np.cos(theta)
# x2 = r*np.sin(theta)

Rp = 1.84*u.Rjup
Rs =  2.36*u.Rsun

Rp_au = Rp.to(u.au).value
Rs_au = Rs.to(u.au).value

DataSetNames = ['2018-06-09','2018-06-18','2019-05-28','2019-06-04']

PhaseLims = np.array([[6.416638102024982970e-01,7.065031027644076422e-01],[2.696165945929052676e-01,4.356450945581210044e-01],[5.617461136892232254e-01,7.118685715981313278e-01],[2.859325695933736533e-01,3.946012406483541679e-01]])
PhaseLimsInRads = PhaseLims*2*np.pi


a_au = 0.03368

x1, x2 = GetXYForCircleArc(0,2*np.pi, 0.03368)

a1, a2 = GetXYForCircleArc(PhaseLimsInRads[0][0],PhaseLimsInRads[0][1], 0.03368*0.8)
b1, b2 = GetXYForCircleArc(PhaseLimsInRads[1][0],PhaseLimsInRads[1][1], 0.03368*0.9)
c1, c2 = GetXYForCircleArc(PhaseLimsInRads[2][0],PhaseLimsInRads[2][1], 0.03368*0.9)
d1, d2 = GetXYForCircleArc(PhaseLimsInRads[3][0],PhaseLimsInRads[3][1], 0.03368*0.8)

Q4x, Q4y = GetXYForCircleArc(0.7210507203934491*2*np.pi,0.8898415065770455*2*np.pi, 0.03368*1.15)
Q1x, Q1y = GetXYForCircleArc(0.11015849342295447*2*np.pi,0.2789492796065509*2*np.pi, 0.03368*1.15)

e1,e2 = GetXYForCircleArc(0,2*np.pi, Rs_au)



fig, ax = plt.subplots(1)


ax.plot(x1, x2,linewidth=1,color='black')

ax.plot(a1, a2,linewidth=5,label=DataSetNames[0])
ax.plot(b1, b2,linewidth=5,label=DataSetNames[1])
ax.plot(c1, c2,linewidth=5,label=DataSetNames[2])
ax.plot(d1, d2,linewidth=5,label=DataSetNames[3])
ax.plot(e1, e2,linewidth=1,color='black')

plt.scatter([0],[a_au],color='black')
plt.scatter([0],[-a_au],color='black')
plt.scatter([-a_au],[0],color='black')
plt.scatter([a_au],[0],color='black')


ax.set_aspect(1)

Plotlims = (-0.05,0.05)

plt.xlim((-0.05,0.06))
plt.ylim((-0.05,0.05))

plt.ylabel('au')
plt.xlabel('au')

plt.text(0,-0.041,'phase = 0\ntransit',horizontalalignment='center',verticalalignment='center')
plt.text(0,+0.041,'phase = 0.5\neclipse',horizontalalignment='center',verticalalignment='center')



plt.text(0,0,'KELT-9',horizontalalignment='center',verticalalignment='center')

plt.legend(loc=4)

# plt.savefig('PhaseCoverageDiagram.png',dpi=400)
plt.savefig('PhaseCoverageDiagram.pdf')


######################
### For proposals 

fig, ax = plt.subplots(1)


ax.plot(x1, x2,linewidth=1,color='black')

ax.plot(a1, a2,linewidth=5,color='grey',label='Archival CARMENES data')
ax.plot(b1, b2,linewidth=5,color='grey')
ax.plot(c1, c2,linewidth=5,color='grey')
ax.plot(d1, d2,linewidth=5,color='grey')
ax.plot(Q1x, Q1y,linewidth=5,label='Proposed Q1')
ax.plot(Q4x, Q4y,linewidth=5,label='Proposed Q4')
ax.plot(e1, e2,linewidth=1,color='black')

plt.scatter([0],[a_au],color='black')
plt.scatter([0],[-a_au],color='black')
plt.scatter([-a_au],[0],color='black')
plt.scatter([a_au],[0],color='black')


ax.set_aspect(1)

Plotlims = (-0.05,0.05)

plt.xlim((-0.06,0.06))
plt.ylim((-0.05,0.08))

plt.ylabel('au')
plt.xlabel('au')

plt.text(0,-0.043,'phase = 0\ntransit',horizontalalignment='center',verticalalignment='center')
plt.text(0,+0.041,'phase = 0.5\neclipse',horizontalalignment='center',verticalalignment='center')



plt.text(0,0,'KELT-9',horizontalalignment='center',verticalalignment='center')

plt.legend(loc=2)

# plt.savefig('PhaseCoverageDiagram.png',dpi=400)
plt.savefig('PhaseCoverageDiagram_NightEmissionProposal.png',dpi=500)

# plt.xlim(-1.25,1.25)
# plt.ylim(-1.25,1.25)

#plt.grid(linestyle='--')

# plt.title('How to plot a circle with matplotlib ?', fontsize=8)

# plt.savefig("plot_circle_matplotlib_01.png", bbox_inches='tight')

# plt.show()












# import numpy as np
# import matplotlib.pyplot as plt 

# #from basic_units import cm
# import numpy as np
# from matplotlib import patches
# import matplotlib.pyplot as plt


# # xcenter, ycenter = 0.38, 0.52
# # width, height = 1e-1, 3e-1
# # angle = -30

# xcenter, ycenter = 0.0, 0.0
# width, height = 1.0, 1.0
# angle = 0


# #theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
# theta = np.deg2rad(np.arange(0.0, 180, 1.0))
# x = 0.5 * width * np.cos(theta)
# y = 0.5 * height * np.sin(theta)

# rtheta = np.radians(angle)
# R = np.array([
#     [np.cos(rtheta), -np.sin(rtheta)],
#     [np.sin(rtheta),  np.cos(rtheta)],
#     ])


# x, y = np.dot(R, np.array([x, y]))
# x += xcenter
# y += ycenter


# fig = plt.figure()
# ax = fig.add_subplot(111, aspect='auto')
# ax.fill(x, y, alpha=0.2, facecolor='yellow',
#         edgecolor='yellow', linewidth=1, zorder=1)

# e1 = patches.Ellipse((xcenter, ycenter), width, height,
#                      angle=angle, linewidth=2, fill=False, zorder=2)

# # ax.add_patch(e1)

# # ax = fig.add_subplot(212, aspect='equal')
# # ax.fill(x, y, alpha=0.2, facecolor='green', edgecolor='green', zorder=1)
# # e2 = patches.Ellipse((xcenter, ycenter), width, height,
# #                      angle=angle, linewidth=2, fill=False, zorder=2)


# # ax.add_patch(e2)