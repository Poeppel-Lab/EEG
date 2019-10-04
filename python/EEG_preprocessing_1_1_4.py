#!/usr/bin/env python 3.6.3
# -*- coding: utf-8 -*-

"""EEG Preprocessing. Enpowered by MNE-Python==0.18.2"""
'''
Update Log:
	2019/03/24
	script_version == 1.1.1
	MNE == 0.17.1
	python == 3.6.3
	Memo: 1. update save step after eog selection
		  2. update new logging function

	2019/06/15
	script_version == 1.1.2
	MNE == 0.17.1
	python == 3.6.3
	Memo: 1. automatic eog channel recognization and dropping, no more manual selection
		  2. bug fix of logging function


    2019/08/05
	script_version == 1.1.3
	MNE == 0.18.2
	python == 3.6.3
	Memo: 1. deprecate read_raw_brainvision function's parameter: event_id
		  2. update events and event_id setting compling with MNE update pack 0.18.2
		  3. some comments update

    2019/08/06
	script_version == 1.1.4
	MNE == 0.18.2
	python == 3.6.3
	Memo: 1. add event_id to logging function

'''

__author__    = "Hao Zhu"
__copyright__ = "Copyright 2019"
__date__      = '2019/08/06'
__license__   = "MIT"
__version__   = "1.1.4"
__email__     = "hz808@nyu.edu"

#################################################################################################################
#IMPORT NEEDED MODULES
#################################################################################################################
import matplotlib 
matplotlib.use('TkAgg')
import mne
#################################################################################################################
# Define logging function
#################################################################################################################
def logging_file(subject_id, epoch_file):
	log_file = open(f'{path}log_file.txt','a')

	if epoch_file.info['bads'] == []:
		interpolated_sensors = None
	else:
		interpolated_sensors = ','.join(epoch_file.info['bads'])

	dropped_epochs = [str(m) for m,n in enumerate(epoch_file.drop_log) if n!=[] and n!=['IGNORED']]
	dropped_ratio = epoch_file.drop_log_stats()

	infos = [f"Subject:{subject_id}\n",f'Event_ID:{epoch_file.event_id}\n',f"Interpolated_sensors:{interpolated_sensors}\n",
	f"Dropped epochs:{','.join(dropped_epochs)}\n",f"Ratio of dropped epochs:{format(dropped_ratio,'.2f')}%\n",
	f"Comments:\n\n"]

	for line in infos:
		log_file.write(line)

	log_file.close()

#################################################################################################################
#set up the file path and name 
#################################################################################################################
# path = 'result/Siqi/EXP4/'
path = 'result/Test/'
name = 'pe002-01' #{.vhdr:loading file .vmrk:trigger file .eeg:raw data file)}
#################################################################################################################
# read raw eeg data 
#################################################################################################################
raw = mne.io.read_raw_brainvision(f'{path}{name}.vhdr',montage='standard_1020',preload=True)
'''
raw: the raw eeg data set --Brainvision == BrainProduct, we use their product of EEG equipment
							path,name: the eeg file path and name
							preload: preload data into memory for data manipulation and faster indexing.
'''

raw_events, raw_event_id = mne.events_from_annotations(raw)
print(raw_event_id)
'''
find events and event_id from raw file automatically
It looks like this:
		raw_events: [[0  0  99999],[14134  0  10001]....]
		raw_event_id: {'New Segment/': 99999, 'Stimulus/BASESOUND': 10001.....}

Rules are flollowings:
	Brainvision: map stimulus events to their i nteger part; 
	response events to integer part + 1000; optic events to integer part + 2000; 
	‘SyncStatus/Sync On’ to 99998; ‘New Segment/’ to 99999; all others like None with an offset of 10000.

Hence we need modification of events and event_id
'''
events = raw_events.copy()[1:]
events[:,2] = raw_events[1:,2] - 10000
#make events last col stays simple as 1,2,3,4

event_id = {m.split('/')[1]:n-10000 for m,n in raw_event_id.items() if m!='New Segment/'}
#make event_id clean

print(event_id)
#################################################################################################################
#preprocessing eeg data
#################################################################################################################
raw = mne.add_reference_channels(raw,'Cz')
# add Cz as the reference channel

raw.set_channel_types(mapping={'AF3': 'eog','AF7': 'eog'})
# set eog channel

raw.set_montage(mne.channels.read_montage('standard_1020'))
# set montage(the standard 10-20 headset) --for topography

raw.pick_types(meg=False, eeg=True, eog=True)
# set pick types, when use eeg and eog

# print(raw.info)
# show raw data information

raw.filter(0.1, 30, n_jobs=2, fir_design='firwin')
# apply band pass filter | n_jobs: Number of jobs to run in parallel | fir_design: “firwin” uses a time-domain design technique that generally gives improved attenuation using fewer samples than “firwin2”

epoch = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True, reject=None)
'''
Epochs extracted from a Raw instance.
									-- raw : Raw object
									   raw._events : pre_created event object that contains events' info
									   event_id: the condition marker that define the starting point of each epoch
									   tmin:  the start of the epoch
									   tmax: T-max, the end of the epoch
									   baseline: The time interval to apply baseline correction. Correction is applied by computing mean of the baseline period and subtracting it from the data.
									   preload: Load all epochs from disk when creating the object or wait before accessing each epoch (more memory efficient but can be slower).
									   reject: Rejection parameters based on peak-to-peak amplitude. set to be None becasue assuming use manul selection
'''
epoch.plot(n_channels=34,block=True,picks=['eeg','eog'])
# plot all epochs across all channels, manully select bad eog epoch by clicking in the matplotlib plot window

epoch.save(f'{path}{name}_step1_backup-epo.fif') 
# save eog artifacts dropped data for backup

eog_ch = [i['ch_name'] for i in epoch.info['chs'] if i['kind']==202]
# pick bad channel names

epoch.drop_channels(eog_ch)
# drop bad channels

epoch.plot(n_channels=32,block=True)
# plot all epochs across all cahnnels again, pick bad channels for interpolation

interpolated_channel = epoch.info['bads']
# record interpolated sensors

if interpolated_channel == []:
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

# epoch.apply_baseline(baseline=(-0.2, 0))
# optional baseline correction application if needed
# if use, please make mne.Epochs(baseline=None) in line 127

epoch.save(f'{path}{name}-epo.fif') 
# save preprocessed eeg epoch data into a .fif file that is easy and efficient to be loaded by further analysis.

epoch.info['bads'] = interpolated_channel
# reset interpolated sensors

logging_file(name,epoch)
# register epoch infos:dropped epochs, interpolated channels
