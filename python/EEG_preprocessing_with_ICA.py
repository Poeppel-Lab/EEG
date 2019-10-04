#################################################################################################################
#IMPORT NEEDED MODULES
#################################################################################################################
import sys
sys.path = ['../']+sys.path
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import easyEEG  
import mne
import os
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs

#################################################################################################################
#set up the file path and name
#################################################################################################################
# path = 'result/Siqi/EXP4/'
path = 'result/Fuyin/patients/'
name = 'pe002T' #{.vhdr:loading file .vmrk:trigger file .eeg:raw data file)}
#################################################################################################################
# read raw eeg data 
#################################################################################################################
# event_id= {'BASETONE':1,'YELLOW_NSD':2,'YELLOW_TONE':3,'YELLOW_SD':6,'BASESOUND':7}
# event_id = {'S': 1,'D': 2,'T': 3,'BASESOUND': 4,'BASETONE': 5,'RTONE': 6,'RSOUND': 7}
event_id = {'AD':1}
# event_id= {'BASETONE':1,'YELLOW_NSD':2,'YELLOW_TONE':3,'YELLOW_SD':4,'BASESOUND':5} #siqi-exp4
# event_id= {'BASETONE':1,'YELLOW_NSD':2,'YELLOW_TONE':3,'YELLOW_SD':4,'BASESOUND':5}#fuyin-exp1
# event_id= {'BASETONE':1,'RED_SAME':6,'RED_DIF':7,'RED_TONE':8,'BASESOUND':5}#fuyin-exp2

raw = mne.io.read_raw_brainvision(f'{path}{name}.vhdr', event_id=event_id, preload=True)
'''
raw: the raw eeg data set --Brainvision == BrainProduct, we use their version of EEG equipment
							pre_path,path,name: the eeg file path and name
							event_id: creat an Epoch.event object that is be used to load all events
							preload: preload data into memory for data manipulation and faster indexing.
'''
##############################################################################################################################
#preprocessing eeg data
##############################################################################################################################
raw = mne.add_reference_channels(raw,'Cz')
# add Cz as the reference channel

raw.set_channel_types(mapping={'AF3': 'eog','AF7': 'eog'})
# set eog channel

raw.set_montage(mne.channels.read_montage('standard_1020'))
# set montage(the standard 10-20 headset) --for topography

raw.drop_channels(['STI 014'])
# drop channel, some default unused channel(like the STI 014)

raw.filter(1.0, 30, n_jobs=2, fir_design='firwin')
# apply band pass filter | n_jobs: Number of jobs to run in parallel | fir_design: “firwin” uses a time-domain design technique that generally gives improved attenuation using fewer samples than “firwin2”

picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,stim=False, exclude='bads')
# pick needed data types

print(raw.info)
# show raw data information

##############################################################################################################################
#ICA
##############################################################################################################################
method = 'fastica'

# Choose other parameters
n_components = 25  # if float, select n_components by explained variance of PCA
decim = 3  # we need sufficient statistics, not all time points -> saves time

# we will also set state of the random number generator - ICA is a
# non-deterministic algorithm, but we want to have the same decomposition
# and the same order of components each time this tutorial is run
random_state = 23

ica = ICA(n_components=n_components, method=method, random_state=random_state)

eeg_reject = dict(eeg=10000000e-6)

ica.fit(raw, picks=picks_eeg, decim=decim, reject=eeg_reject)

eog_average = create_eog_epochs(raw, reject=eeg_reject, picks=picks_eeg).average() # get averaged EOG trial

eog_epochs = create_eog_epochs(raw, reject=eeg_reject)  # get single EOG trials

eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation

print(ica.labels_)

##############################################################################################################################
#mute these two lines first, find right components then fill the list
# eog_inds = [0,1,2,3,5] #This is where you change components 

# ica.labels_ = {'eog/0/AF7': eog_inds, 'eog/1/AF3': eog_inds, 'eog': eog_inds}

##############################################################################################################################
#ICA plots
ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components

ica.plot_sources(eog_average, exclude=eog_inds)  # look at source time course

ica.plot_components() #look at the component topography

ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 35.}, image_args={'sigma': 1.}) # look at the combination of all infos for selected component

icapoltoverlay = ica.plot_overlay(eog_average, exclude=eog_inds, show=False) # plot ERP after selected components are removed
##############################################################################################################################
#apply ICA
ica.exclude.extend(eog_inds) #add selected components into ica calss
 
print(ica.labels_)

ica.apply(raw) # apply ICA

##############################################################################################################################
#Rest preprocessing steps after applied ICA
##############################################################################################################################
epoch = mne.Epochs(raw, raw._events, event_id, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True, reject=None)
# '''
# Epochs extracted from a Raw instance.
# 									-- raw : Raw object
# 									   raw._events : pre_created event object that contains events' info
# 									   event_id: the condition marker that define the starting point of each epoch
# 									   tmin:  the start of the epoch
# 									   tmax: T-max, the end of the epoch
# 									   baseline: The time interval to apply baseline correction. Correction is applied by computing mean of the baseline period and subtracting it from the data.
# 									   preload: Load all epochs from disk when creating the object or wait before accessing each epoch (more memory efficient but can be slower).
# 									   reject: Rejection parameters based on peak-to-peak amplitude. set to be None becasue assuming use manul selection
# '''
epoch.plot(n_channels=34,block=True)
# # plot all epochs across all channels, manully select bad eog epoch and delete eog channels by clicking in the matplotlib plot window

bad_list = epoch.info['bads']  
# pick bad channel names

epoch.drop_channels(bad_list)
# drop bad channels

epoch.plot(n_channels=32,block=True)
# plot all epochs across all cahnnels again, pick bad channels for interpolation

if epoch.info['bads'] == []:
	# if no channel needs to be interpolated, pass
	pass
else:
	epoch.interpolate_bads(reset_bads=True, mode='accurate', verbose=None)
	# interpolate bad channels

	epoch.plot(n_channels=32,block=True)
	# see epoch plot after interpolating bad channels

epoch.set_eeg_reference(ref_channels='average', projection=True)
# set EEG average reference,re_reference the data, all data across the channels addsup to be 0
# In case of (ref_channels='average') in combination with (projection=True), the reference is added as a projection and it is not applied automatically. 
# For it to take effect, apply with method apply_proj

epoch.apply_proj()
# apply re-reference projections.

eeg_reject = dict(eeg=100e-6)
# set up rejection range
 
epoch.drop_bad(reject=eeg_reject, flat=None, verbose=None)
# automatically drop bad epochs based on defined rejection criteria

epoch.plot(n_channels=32,block=True)
# plot final epoch result, for checking purpose 

epoch.save(f'{path}{name}-epo.fif') 
# save preprocessed eeg epoch data into a .fif file that is easy and efficient to be loaded by further analysis.
