
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import publication_settings
import pandas as pd

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

matplotlib.rcParams.update(publication_settings.params)

# ** define CSV file **
# data load
data = pd.read_csv('OS_train_R_SDPOD.csv')
OS_R_SDPOD = np.array(data['OS'])
OScensor_R_SDPOD = np.array(data['OS_censor'])

data = pd.read_csv('OS_train_R_CRPR.csv')
OS_R_CRPR = np.array(data['OS'])
OScensor_R_CRPR = np.array(data['OS_censor'])

data = pd.read_csv('OS_train_mlR_SDPOD.csv')
OS_mlR_SDPOD = np.array(data['OS'])
OScensor_mlR_SDPOD = np.array(data['OS_censor'])

data = pd.read_csv('OS_train_mlR_CRPR.csv')
OS_mlR_CRPR = np.array(data['OS'])
OScensor_mlR_CRPR = np.array(data['OS_censor'])

# ** define months 0, 0.1, 0.2... max(PFS) **
#months = np.linspace(0,np.max(PFS),int(10*np.max(PFS))+1)
months = np.linspace(0,np.max(OS_R_CRPR),int(10*np.max(OS_R_CRPR))+1)
prob_of_survival = np.zeros(len(months))
num_at_risk = np.zeros(len(months))

def KaplanMeier(aDuration,aCensor):
	
	mask_censor = (aCensor==0)
	censored_duration = aDuration[mask_censor]

	num_alive = len(aCensor)
	prob_of_survival[0] = 1.

	for j,month in enumerate(months):
		num_died = np.sum( (np.abs(aDuration-month)<0.01) & (aCensor==1) )
		num_censored = np.sum( (np.abs(aDuration-month)<0.01) & (aCensor==0) )

		if num_died == 0 and j>0:
			prob_of_survival[j] = prob_of_survival[j-1]
			num_alive -= num_censored

		if num_died > 0 and j>0:
		
			prob_of_survival[j] = prob_of_survival[j-1]*((num_alive-num_died)
									/(num_alive))
			num_alive = num_alive - num_died - num_censored
			
		num_at_risk[j] = num_alive

	return months,prob_of_survival,censored_duration,len(aCensor),num_at_risk,aDuration,aCensor

def hazard_ratio_censor(group1,group2,group1censor,group2censor):

	#change PFS_censor != 0 if calc PFS hazard ratios
	deaths1 = np.sum(group1censor)
	deaths2 = np.sum(group2censor)

	mo = np.arange(0,np.ceil(max(np.max(group1),np.ceil(np.max(group2)))))

	expected_deaths1 = 0
	expected_deaths2 = 0

	for i,month in enumerate(mo):
		deaths1_mo_x = np.sum( (group1 < (month+1)) & (group1 >= month) & (group1censor == 1) )
		alive1_mo_x = np.sum( group1 >= month )
		deaths2_mo_x = np.sum( (group2 < (month+1)) & (group2 >= month) & (group2censor == 1) )
		alive2_mo_x = np.sum( group2 >= month )

		prob_death_mo_x = (deaths1_mo_x+deaths2_mo_x)/(alive1_mo_x+alive2_mo_x)

		expected_deaths1 += prob_death_mo_x*alive1_mo_x
		expected_deaths2 += prob_death_mo_x*alive2_mo_x

	hazard_ratio=(deaths1/expected_deaths1)/(deaths2/expected_deaths2)

	se = np.sqrt(1/expected_deaths1 + 1/expected_deaths2)
	hazard_ratio_m = hazard_ratio*np.exp(-1.96*se)
	hazard_ratio_p = hazard_ratio*np.exp(+1.96*se)

	return hazard_ratio_m, hazard_ratio, hazard_ratio_p

# ******* DESIGN *******
# ******* DESIGN *******
# 8.5 x 5 inches
total_width = 5
total_height = 4.5
fig = plt.figure(1, figsize=(total_width,total_height))

# give all in inches
top_border = 0.2
bottom_border = 1.4
left_border = 1.5
right_border = 0.1

left = left_border/total_width
bottom = bottom_border/total_height
width = (total_width - left_border - right_border)/total_width
height = (total_height - top_border - bottom_border)/total_height

ax = fig.add_axes([left,bottom,width,height])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

y_max = 1.01
x_max = 72
ax.set_xlim([0,48])
ax.set_ylim([0,y_max])

ax.set_yticks([0,.25,.50,.75,1])
ax.set_yticklabels((r"0%",r"25%",r"50%",r"75%",r"100%"))

ax.set_xticks([0,12,24,36,48])

font = 'Arial'

ax.set_ylabel(r"overall survival (%)",fontname=font)
ax.set_xlabel('months elapsed (months)',fontname=font)

grey = '#919191' # grey

purp6 = '#807dba' #purp6
purp7 = '#6a51a3' #purp7
purp8 = '#4a1486' #purp8

#green6 = '#41ab5d' #gr6
#green7 = '#238b45' #gr7
#green8 = '#005a32' #gr8

green6 = '#4292c6' #blue6
green7 = '#2171b5' #blue7
green8 = '#084594' #blue8

# ******* DESIGN *******
# ******* DESIGN *******

ax.text(-20,-0.14, 'No. at risk', fontweight='bold', horizontalalignment="left"
	,verticalalignment="center",fontsize=10,fontname=font,clip_on=False)

nor_times = np.array([0,12,24,36,48])
nor_times10 = nor_times*10

# graph KM curve 
#OS_R_SDPOD = np.array(data['OS'])
#OScensor_R_SDPOD = np.array(data['OS_censor'])

m_SD20P3,pos_SD20P3,c_PFS_SD20P3,t,n_SD20P3,dur_SD20P3,censor1 = KaplanMeier(OS_R_SDPOD,OScensor_R_SDPOD)
ax.plot(m_SD20P3, pos_SD20P3,color=purp6,linewidth=2.)
		
for value in c_PFS_SD20P3:
	i = np.argmin(np.abs(months-value))
	ax.scatter(value,pos_SD20P3[i],marker='|',color=purp6,zorder=4,s=30)

n_SD20P3_times = n_SD20P3[nor_times10]

ax.text(-20,-0.26, "RECIST SD/POD", color=purp6,fontweight='bold', horizontalalignment="left"
	,verticalalignment="center",fontsize=10,fontname=font,clip_on=False)

for i,(nor_times_i,n_SD20P3t_i) in enumerate(zip(nor_times,n_SD20P3_times)):
  ax.text(nor_times_i,-0.26, int(n_SD20P3t_i), color=purp6, horizontalalignment="center"
  	,verticalalignment="center",fontsize=10,fontname=font,clip_on=False)

# graph KM curve
#OS_R_CRPR = np.array(data['OS'])
#OScensor_R_CRPR = np.array(data['OS_censor'])
m_PRmedGreat,pos_PRmedGreat,c_PFS_PRmedGreat,t,n_PRmedGreat,dur_PRmedGreat,censor1 = KaplanMeier(OS_R_CRPR,OScensor_R_CRPR)
ax.plot(m_PRmedGreat, pos_PRmedGreat,color=green6,linewidth=2.)
		
for value in c_PFS_PRmedGreat:
	i = np.argmin(np.abs(months-value))
	ax.scatter(value,pos_PRmedGreat[i],marker='|',color=green6,zorder=4,s=30)

n_PRmedGreat_times = n_PRmedGreat[nor_times10]

ax.text(-20,-0.18, "RECIST CR/PR", color=green6,fontweight='bold', horizontalalignment="left"
	,verticalalignment="center",fontsize=10,fontname=font,clip_on=False)

for i,(nor_times_i,n_PRmedGreatt_i) in enumerate(zip(nor_times,n_PRmedGreat_times)):
  ax.text(nor_times_i,-0.18, int(n_PRmedGreatt_i), color=green6, horizontalalignment="center"
  	,verticalalignment="center",fontsize=10,fontname=font,clip_on=False)

# graph KM curve
#OS_mlR_SDPOD = np.array(data['OS'])
#OScensor_mlR_SDPOD = np.array(data['OS_censor'])
m_PRmedGreat,pos_PRmedGreat,c_PFS_PRmedGreat,t,n_PRmedGreat,dur_PRmedGreat,censor1 = KaplanMeier(OS_mlR_SDPOD,OScensor_mlR_SDPOD)
ax.plot(m_PRmedGreat, pos_PRmedGreat,color=purp8,linewidth=2.5,zorder=4,linestyle='--')
		
for value in c_PFS_PRmedGreat:
	i = np.argmin(np.abs(months-value))
	ax.scatter(value,pos_PRmedGreat[i],marker='|',color=purp8,zorder=4,s=30)

n_PRmedGreat_times = n_PRmedGreat[nor_times10]

ax.text(-20,-0.3, "mlRECIST SD/POD", color=purp8,fontweight='bold', horizontalalignment="left"
	,verticalalignment="center",fontsize=10,fontname=font,clip_on=False)

for i,(nor_times_i,n_PRmedGreatt_i) in enumerate(zip(nor_times,n_PRmedGreat_times)):
  ax.text(nor_times_i,-0.3, int(n_PRmedGreatt_i), color=purp8, horizontalalignment="center"
  	,verticalalignment="center",fontsize=10,fontname=font,clip_on=False)

# graph KM curve
#OS_mlR_CRPR = np.array(data['OS'])
#OScensor_mlR_CRPR = np.array(data['OS_censor'])
m_PRmedGreat,pos_PRmedGreat,c_PFS_PRmedGreat,t,n_PRmedGreat,dur_PRmedGreat,censor1 = KaplanMeier(OS_mlR_CRPR,OScensor_mlR_CRPR)
ax.plot(m_PRmedGreat, pos_PRmedGreat,color=green8,linewidth=2.5,zorder=4,linestyle='--')
		
for value in c_PFS_PRmedGreat:
	i = np.argmin(np.abs(months-value))
	ax.scatter(value,pos_PRmedGreat[i],marker='|',color=green8,zorder=4,s=30)

n_PRmedGreat_times = n_PRmedGreat[nor_times10]

ax.text(-20,-0.22, "mlRECIST CR/PR", color=green8,fontweight='bold', horizontalalignment="left"
	,verticalalignment="center",fontsize=10,fontname=font,clip_on=False)

for i,(nor_times_i,n_PRmedGreatt_i) in enumerate(zip(nor_times,n_PRmedGreat_times)):
  ax.text(nor_times_i,-0.22, int(n_PRmedGreatt_i), color=green8, horizontalalignment="center"
  	,verticalalignment="center",fontsize=10,fontname=font,clip_on=False)

# make legend


plt.savefig('mlR_F2_KM_mlR_train.png',dpi=300)
plt.savefig('mlR_F2_KM_mlR_train.eps')