
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import publication_settings
import pandas as pd

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve

matplotlib.rcParams.update(publication_settings.params)

#data = pd.read_csv('auc_trainingPFS.csv')
data = pd.read_csv('auc_training.csv')
prob_A = np.array(data['Prediction_A'])
prob_B = np.array(data['Prediction_B'])
prob_C = np.array(data['Prediction_C'])
truth = np.array(data['Truth'])

fprA, tprA, threshholdA = roc_curve(truth, prob_A)
aucA = roc_auc_score(truth, prob_A)
fprB, tprB, threshholdB = roc_curve(truth, prob_B)
aucB = roc_auc_score(truth, prob_B)
fprC, tprC, threshholdC = roc_curve(truth, prob_C)
aucC = roc_auc_score(truth, prob_C)

# 5 x 5 inches
total_width = 5
total_height = 5
fig = plt.figure(1, figsize=(total_width,total_height))

# give all in inches
top_border = 0.5
bottom_border = 0.7
left_border = 0.7
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

#y_unit = total_height*height/y_max
#x_unit = total_width*width/num_entries

font = 'Arial'

ax.set_ylabel('sensitivity',fontname=font,fontsize=12)
ax.set_xlabel('1 - specificity',fontname=font,fontsize=12)
ax.set_xlim([0,1.02])
ax.set_ylim([0,1.02])
#ax.set_yticks([0,.1,0.2,0.3,0.4,0.5])
#ax.set_yticklabels((r"0%",r"10%",r"20%",r"30%",r"40%",r"50%"),fontsize=12)
#ax.set_xticks([-6,-4,-2,0,2,4,6])

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

#blue6 = '#807dba' #purp6
#blue7 = '#6a51a3' #purp7
#blue8 = '#4a1486' #purp8

#fpr, tpr, threshhold = roc_curve(truth, prob_A)
#auc = roc_auc_score(truth, prob_A)

ax.plot(ax.get_xlim(), ax.get_ylim(), color= grey, linewidth=2.5)

ax.plot(fprA, tprA ,color=blue8,linewidth=3,zorder=4)
ax.plot(fprB, tprB ,color=blue7,linewidth=2.5, linestyle='--',zorder=1)
ax.plot(fprC, tprC ,color=blue6,linewidth=2.5, linestyle='--',zorder=1)

AUC_a = "Method A AUC = %.2f" % aucA
AUC_b = "Method B AUC = %.2f" % aucB
AUC_c = "Method C AUC = %.2f" % aucC

ax.text(0.5,0.3, AUC_a, horizontalalignment="left",verticalalignment="center",
	fontsize=12,fontname=font, color=blue8)
ax.text(0.5,0.25, AUC_b, horizontalalignment="left",verticalalignment="center",
	fontsize=12,fontname=font, color=blue7)
ax.text(0.5,0.2, AUC_c, horizontalalignment="left",verticalalignment="center",
	fontsize=12,fontname=font, color=blue6)

#plt.savefig('mlR_F_ROC_trainORR.png',dpi=300)
plt.savefig('mlR_F_ROC_trainORR.eps')

#plt.savefig('mlR_F_ROC_trainPFS.png',dpi=300)
#plt.savefig('mlR_F_ROC_trainPFS.eps')