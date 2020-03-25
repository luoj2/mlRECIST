import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import publication_settings
import pandas as pd

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

matplotlib.rcParams.update(publication_settings.params)

data = pd.read_csv('ensemble_vs_truth.csv')

y_val = np.array(data['Ensemble'])
x_val = np.array(data['Truth'])

# 5 x 5 inches
total_width = 4
total_height = 4
fig = plt.figure(1, figsize=(total_width,total_height))

# give all in inches
top_border = 0.5
bottom_border = 0.6
left_border = 0.6
right_border = 0.5

left = left_border/total_width
bottom = bottom_border/total_height
width = (total_width - left_border - right_border)/total_width
height = (total_height - top_border - bottom_border)/total_height

ax = fig.add_axes([left,bottom,width,height])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
#ax.axhline(y=0,xmin=0,xmax=1,color='k',zorder=4)
#ax.axhline(y=20,xmin=0,xmax=1,linestyle='--',color='k',zorder=0)
#ax.axhline(y=-30,xmin=0,xmax=1,linestyle='--',color='k',zorder=0)

y_max = 26
y_min = 0
x_min = 0
x_max = 26

y_unit = total_height*height/y_max
x_unit = total_width*width/x_max

#for xlabel_i in ax.get_xticklabels():
 #   xlabel_i.set_visible(False)

#minor_ticks = np.arange(0, num_entries, 1)
#ax.set_yticks([])
#ax.tick_params('y',direction='out',which='both')
#ax.set_yticks(minor_ticks,minor=True)

ax.set_xlim([x_min,x_max])
ax.set_ylim([y_min,y_max])

ax.set_yticks([0,6,12,18,24])
ax.set_xticks([0,6,12,18,24])

font = 'Arial'

ax.set_ylabel(r"mlRECIST PFS (months)",fontname=font,fontsize=14)
ax.set_xlabel('RECIST PFS (months)',fontname=font,fontsize=14)

grey = '#919191' # grey

turq = '#1b9e77' # turq
orange = '#d95f02' # orange
purp = '#7570b3' # purple
pink = '#e7298a' # pink
green = '#66a61e' # green
tan = '#e6ab02' # tan
brown = '#a6761d' # brown
dkgrey = '#666666' # dk grey

blue1 = '#f7fbff' #blue1
blue2 = '#deebf7' #blue2
blue3 = '#c6dbef' #blue3

blue4 = '#9ecae1' #blue4
blue5 = '#6baed6' #blue5
blue6 = '#4292c6' #blue6
blue7 = '#2171b5' #blue7
blue8 = '#084594' #blue8

purp8 = '#4a1486'

ax.scatter(x_val,y_val,color=purp8,marker='o',s=30,edgecolors=grey,alpha=0.5)

ax.text(3,20, "r=0.83, p<0.001", horizontalalignment="left",verticalalignment="center",
	fontsize=12,fontname=font)

plt.savefig('mlR_F_scatter_trainPFS.png',dpi=300)
#plt.savefig('mlR_F_scatter_trainPFS.eps')