
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import publication_settings
import pandas as pd

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

matplotlib.rcParams.update(publication_settings.params)

categories = 5
labels = ['CR/PR','SD/POD','Y/N','Exact date','+/- 2 months']

data = pd.read_csv('accuracy_train_valid.csv')

train = np.array(data['Training'])
valid = np.array(data['Validation'])
ext = np.array(data['Ext'])

# 5 x 8 inches
total_width = 8
total_height = 5
fig = plt.figure(1, figsize=(total_width,total_height))

# give all in inches
top_border = 0.5
bottom_border = 0.7
left_border = 1
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

ax.set_xlim([0,4.2])
ax.set_ylim([0,1])

font = 'Arial'

ax.set_yticks([0,.25,.5,.75,1])
ax.set_yticklabels((r"0%",r"25%",r"50%",r"75%",r"100%"),fontsize=12)
ax.set_ylabel('% Accuracy Compared to RECIST',fontname=font,fontsize=12)

x=np.array([0.4,1.2,2,2.8,3.6])
width = 0.2
ax.set_xticks(x)
ax.set_xticklabels((labels), fontsize=12)
ax.xaxis.set_tick_params(length=0)

blue6 = '#4292c6' #blue6
blue7 = '#2171b5' #blue7
blue8 = '#084594' #blue8

green5 = '#74c476' #gr5
green6 = '#41ab5d' #gr6
green7 = '#238b45' #gr7
green8 = '#005a32' #gr8

pink6 = '#dd3497' #pink6
pink7 = '#ae017e' #pink7
pink8 = '#7a0177' #pink8

print(train,valid)

Train_bar = ax.bar(x-0.2, train, width, color=blue6,edgecolor='k')
Valid_bar = ax.bar(x, valid, width, color=green5,edgecolor='k')
Ext_bar = ax.bar(x+0.2, ext, width, color=pink8,edgecolor='k')

fig.tight_layout()

#for i,(PFS,censor) in enumerate(zip(PFS_list,censor_list)):

#  j = num_entries - i - 1
  # make box
#  ax.add_patch( Rectangle([0,j+0.1],PFS,0.8,facecolor=blue) )

# make legend
legend_x = 2.55
legend_y = 1
legend_dy = 0.03
x_max=2
legend_dx = legend_dy/categories*(height*total_height)*x_max/(width*total_width)
#
delta_y = 0.06
one = "Training (n=361)"
two = "Validation (n=92)"
three = "External validation (n=97)"
labels = [one, two, three]
colors = [blue6,green5,pink8]
#labels = [two, three]
for i,(color,label) in enumerate(zip(colors,labels)):
  #ax.add_patch( Rectangle([legend_x,legend_y-i*delta_y],legend_dx,legend_dy,facecolor=color,
  # edgecolor='w') )
  ax.scatter(legend_x,legend_y-i*delta_y+ legend_dy/2,color=color,marker='s',s=100,clip_on=False)
  ax.text( legend_x + legend_dx + 0.1,legend_y - i*delta_y + legend_dy/2, label, 
    horizontalalignment="left",verticalalignment="center",fontsize=12,fontname=font)

#plt.savefig('mlR_F_accuracy_trainvalidext.png',dpi=300)
plt.savefig('mlR_F_accuracy_trainvalidext.eps')

