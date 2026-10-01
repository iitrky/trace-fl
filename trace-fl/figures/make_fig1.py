"""Draw Fig. 1 of the paper (system model and TRACE-FL+ server pipeline).

  python figures/make_fig1.py   ->  figures/fig1_system_model.png (300 dpi)
"""
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family']='DejaVu Serif'
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 110); ax.set_ylim(0, 62); ax.axis('off')
def box(x,y,w,h,txt,fc,ec='#333',fs=9,bold=False,r=1.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.2,rounding_size={r}",fc=fc,ec=ec,lw=1.1))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,weight='bold' if bold else 'normal',wrap=True)
def arrow(x1,y1,x2,y2,c='#333',ls='-',txt=None,tx=0,ty=0,rad=0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,color=c,lw=1.2,ls=ls,connectionstyle=f"arc3,rad={rad}"))
    if txt: ax.text((x1+x2)/2+tx,(y1+y2)/2+ty,txt,fontsize=8,ha='center',color=c,style='italic')
# clients column
ax.text(9,59.5,'Clients (non-IID data)',ha='center',fontsize=10,weight='bold')
ys=[49,40,31,22,13]
labels=[('Honest client 1\n$D_1$, local SGD','#DCEBDD'),('Honest client 2\n$D_2$, local SGD','#DCEBDD'),('Byzantine client\n(ALIE / IPM / SF …)','#F6D5D5'),('Honest client $n-1$\n$D_{n-1}$','#DCEBDD'),('Byzantine client\n(adaptive A-IPM …)','#F6D5D5')]
for y,(t,c) in zip(ys,labels): box(1,y,16,6.5,t,c,fs=8)
ax.text(9,57.2,'$f=\\rho n$ Byzantine, $\\rho<0.5$',ha='center',fontsize=8)
# server frame
ax.add_patch(FancyBboxPatch((25,3),83,55,boxstyle="round,pad=0.3,rounding_size=2",fc='#F7F9FC',ec='#4A6FA5',lw=1.5,ls='--'))
ax.text(66.5,55.6,'Server: TRACE-FL+ aggregation (round $t$)',ha='center',fontsize=10.5,weight='bold',color='#23426E')
for y in ys: arrow(17.4,y+3.25,27,32,c='#555')
ax.text(21.5,50,'updates\n$u_i^t$',fontsize=8,ha='center',style='italic')
S='#E3ECF8'
box(27,28,13,8,'Step 1\nMedian-norm\nclipping $\\tilde u_i$',S,fs=8.5)
box(43.5,28,14,8,'Step 2\nReference $m_t$\n(coord. median)',S,fs=8.5)
arrow(40.4,32,43.1,32)
box(61,18,25,26,'',"#FFF6E0",ec='#B8860B')
ax.text(73.5,41.2,'Step 3: trust $q_i^t$',ha='center',fontsize=9,weight='bold')
for k,t in enumerate(['$c_i$  direction consistency','$\\delta_i$  robust distance (MAD)','$\\kappa_i$  collusion (NN similarity)','$\\pi_i$  trajectory persistence']):
    ax.text(62.3,37.3-k*3.7,t,fontsize=8,va='center')
ax.plot([62.5,84.5],[23.4,23.4],color='#B8860B',lw=0.6)
ax.text(73.5,20.6,'$q_i^t=c_i\\,\\delta_i\\,\\kappa_i\\,\\pi_i$',ha='center',fontsize=9)
arrow(57.9,32,60.6,32)
box(89.5,27,17,10,'Step 4\nTemporal reputation\n$r_i^t=\\gamma r_i^{t-1}$\n$+(1-\\gamma)\\,q_i^t$',S,fs=8)
arrow(86.4,32,89.1,32)
box(89.5,9,17,10,'Step 5 (TRACE-FL+)\nHistory weight $a_i$ on\n$H_i=\\Sigma_\\tau \\tilde u_i^\\tau$',S,fs=8)
box(61,7,25,8.5,'Step 6: weighted aggregation\n$w_i=r_i\\,a_i,\\;\\; g_t=\\Sigma_i w_i\\tilde u_i\\,/\\,\\Sigma_i w_i$','#D6E9D8',ec='#2E7D32',fs=8)
arrow(98,26.6,98,19.4)
arrow(89.1,13,86.4,11.5)
box(27,12,21.5,9,'Server state per client\n$r_i$,  $D_i^{prev}$,  $H_i$\n(memory $O(nd)$, Steps 3\u20135)','#EEEEEE',fs=8)
arrow(48.9,18,60.6,21,c='#777',ls='--')
ax.plot([73.5,9],[5.2,5.2],color='#2E7D32',lw=1.3)
arrow(9,5.2,9,12.6,c='#2E7D32')
ax.plot([73.5,73.5],[6.6,5.2],color='#2E7D32',lw=1.3)
ax.text(40,6.2,'broadcast  $w_{t+1}=w_t+g_t$',fontsize=8.5,ha='center',color='#2E7D32',style='italic')
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig1_system_model.png'),dpi=300,bbox_inches='tight',facecolor='white')
