import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from custom_signal_process  import RadarSignalP7
import sys
from DBReader.DBReader import SyncReader
from joblib import dump, load
# this script is used to create the dataset WITH FIRST FFT PLUS SECOND FFT

SAVE_SAMPLES=True

sequence = 'RECORD@2020-11-22_12.08.31'
root_folder=f'/home/christophe/RADIalP7/DATASET/{sequence}'

if not os.path.exists(root_folder):
    print("Folder does not exist")
    raise Exception("Folder does not exist")

#save_folder=f'/media/christophe/backup/DATARADIAL/{sequence}'

# uncomment after test
#save_folder=f'/home/christophe/RADIalP7/SMALL_DATASET/{sequence}'

print('warning just for tests!!!')
save_folder=f'/home/christophe/RADIalP7/SMALL_DATASET/TEST/'

adc_folder=save_folder+'/ADC/'
fft_folder=save_folder+'/FFT/'
fft_2=save_folder+'/FFT2/'


if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    os.makedirs(adc_folder)
    os.makedirs(fft_folder)
    os.makedirs(fft_2)
    print('succesfully created folders where data will be saved')
else:
    if SAVE_SAMPLES:
        sys.exit('Warning sequence seems to have been already computed')

db = SyncReader(root_folder)

print('elements found in sequence parsed: ',len(db))

calib_path='/home/christophe/RADIalP7/SignalProcessing/CalibrationTable.npy'
RSP = RadarSignalP7(path_calib_mat=calib_path,method='RD',device='cpu')
print('will build: ',len(db),'range doppler plus raw data elements')

hanning_window_range=RSP.get_window_hanning_range()
if SAVE_SAMPLES:
    save_hanning=os.path.join(save_folder,f'hanning_window_range.npy')
    np.save(save_hanning,hanning_window_range)

hanning_window_dopller=RSP.get_window_hanning_dopller()
if SAVE_SAMPLES:
    save_hanning=os.path.join(save_folder,f'hanning_window_dopller.npy')
    np.save(save_hanning,hanning_window_dopller)


limit_sample=29
for i in range (len(db)):
    
    sample = db.GetSensorData(i)
    
    raw_adc=RSP.get_raw_adc(sample['radar_ch0']['data'],sample['radar_ch1']['data'],sample['radar_ch2']['data'],sample['radar_ch3']['data'])
    
    save_adc_path=os.path.join(adc_folder,f'raw_adc_{i}.npy')
    if SAVE_SAMPLES:
        np.save(save_adc_path,raw_adc)

    first_fft_map=RSP.get_first_fft(sample['radar_ch0']['data'],sample['radar_ch1']['data'],sample['radar_ch2']['data'],sample['radar_ch3']['data'])
    first_fftV2=RSP.get_first_fftV2(raw_adc)
    assert np.allclose(first_fft_map, first_fftV2)==True, "values differ"

    fft_by_matrix=RSP.build_fft_by_dot_product(raw_adc)

    assert np.allclose(first_fft_map,fft_by_matrix)==True, "DFT differ"

    matrix_dft=RSP.build_fft_matrix(raw_adc)


    save_fft_path=os.path.join(fft_folder,f'first_fft_{i}.npy')
    if SAVE_SAMPLES:
        np.save(save_fft_path,fft_by_matrix)

    
    second_fft=RSP.compute_second_fft(fft_by_matrix)
    save_path2=os.path.join(fft_2,f'second_fft_{i}.npy')
    if SAVE_SAMPLES:
        np.save(save_path2,second_fft)

    print(i)
    if i>limit_sample:
        break   

print('end of the script')

