import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
from custom_signal_process  import RadarSignalP7
import sys
from DBReader.DBReader import SyncReader
from joblib import dump, load
# this script is used to create the dataset WITH FIRST FFT PLUS SECOND FFT

SAVE_ADC=False
SAVE_RANGE_FFT=False
SAVE_RD=False
SAVE_WINDOW=False
SAVE_IMG=False
save_labels=False

Benchmark_Time=False
if Benchmark_Time:
    assert not SAVE_ADC and not SAVE_RANGE_FFT and not SAVE_RD and not SAVE_WINDOW, "data not saved while benchmarking"

## sequence inititialy used in STREAM 
sequence = 'RECORD@2020-11-21_13.57.07'

#sequence='RECORD@2020-11-21_11.54.31'
print('BEWARE OF SEQUENCES!!!')
root_folder=f'/home/christophe/RADIalP7/DATASET/{sequence}'

#df_labels=pd.read_csv(f'/home/christophe/RADIalP7/SignalProcessing/stream_labels_{sequence}.csv')
labels = pd.read_csv('/home/christophe/ComplexNet/STREAM/labels_CVPR.csv')
records = np.unique(labels['dataset'])[:1]

print(records)

sys.exit()






print(df_labels.columns)
print('Number of BBOXES: ')
print(df_labels.shape[0])

if not os.path.exists(root_folder):
    print("Folder does not exist")
    raise Exception("Folder does not exist")

#save_folder=f'/media/christophe/backup/DATARADIAL/{sequence}'


save_folder=f'/home/christophe/RADIalP7/STREAM/'

# print('warning just for tests!!!')
# save_folder=f'/home/christophe/RADIalP7/SMALL_DATASET/VIDEO/'

adc_folder=save_folder+'/ADC/'
fft_folder=save_folder+'/FFT/'
fft2_folder=save_folder+'/FFT2/'
image_folder=save_folder+'/IMG'
label_folder=save_folder+'/LABELS'


if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    os.makedirs(adc_folder)
    os.makedirs(fft_folder)
    os.makedirs(fft2_folder)
    os.makedirs(image_folder)
    os.makedirs(label_folder)
    print('succesfully created folders where data will be saved')
else:
    if SAVE_ADC or SAVE_IMG or SAVE_RANGE_FFT or SAVE_RD or SAVE_WINDOW or save_labels:
        sys.exit('Warning sequence seems to have been already computed')

db = SyncReader(root_folder)

print('elements found in sequence parsed: ',len(db))

calib_path='/home/christophe/RADIalP7/SignalProcessing/CalibrationTable.npy'
RSP = RadarSignalP7(path_calib_mat=calib_path,method='RD',device='cpu')
print('will build: ',len(db),'range doppler plus raw data elements')

hanning_window_range=RSP.get_window_hanning_range()
if SAVE_WINDOW:
    save_hanning=os.path.join(save_folder,f'hanning_window_range.npy')
    np.save(save_hanning,hanning_window_range)

hanning_window_dopller=RSP.get_window_hanning_dopller()
if SAVE_WINDOW:
    save_hanning=os.path.join(save_folder,f'hanning_window_dopller.npy')
    np.save(save_hanning,hanning_window_dopller)


limit_sample=100000
count=0
time_first_fft=[]
time_first_fft_v2=[]
time_first_fft_torch=[]
time_second_fft_torch=[]
time_second_fft=[]
ratio_frame=0

for i in range (len(db)):
    
    sample = db.GetSensorData(i)
    
    image=sample['camera']['data']
    if SAVE_IMG:
        save_img_path=os.path.join(image_folder,f'img_{i}.npy')
        np.save(save_img_path,arr=image)

    raw_adc=RSP.get_raw_adc(sample['radar_ch0']['data'],sample['radar_ch1']['data'],sample['radar_ch2']['data'],sample['radar_ch3']['data'])
    
    save_adc_path=os.path.join(adc_folder,f'raw_adc_{i}.npy')
    if SAVE_ADC:
        np.save(save_adc_path,raw_adc)
    first_fft_start=time.time()
    first_fft_map=RSP.get_first_fft(sample['radar_ch0']['data'],sample['radar_ch1']['data'],sample['radar_ch2']['data'],sample['radar_ch3']['data'])
    time_first_fft.append(time.time()-first_fft_start)

    first_fft_start_v2=time.time()
    first_fftV2=RSP.get_first_fftV2(raw_adc)
    time_first_fft_v2.append(time.time()-first_fft_start_v2)

    assert np.allclose(first_fft_map, first_fftV2)==True, "FFT differ between first and second FFT"



    fft_by_matrix=RSP.build_fft_by_dot_product(raw_adc)

    assert np.allclose(first_fft_map,fft_by_matrix)==True, "DFT differ"
    fft_torch_time=time.time()
    first_fft_torch=RSP.get_torch_first_fft(sample['radar_ch0']['data'],sample['radar_ch1']['data'],sample['radar_ch2']['data'],sample['radar_ch3']['data'])
    time_first_fft_torch.append(time.time()-fft_torch_time)

    assert np.allclose(first_fft_map,first_fft_torch), "FFT differ between radial and torch implementation"

    #
    matrix_dft=RSP.build_fft_matrix(raw_adc)


    save_fft_path=os.path.join(fft_folder,f'first_fft_{i}.npy')
    if SAVE_RANGE_FFT:
        np.save(save_fft_path,fft_by_matrix)

    second_fft_start=time.time()
    second_fft=RSP.compute_second_fft(fft_by_matrix)
    time_second_fft.append(time.time()-second_fft_start)

    second_fft_torch_start=time.time()
    second_fft_torch=RSP.compute_second_fft_torch(fft_by_matrix)
    time_second_fft_torch.append(time.time()-second_fft_torch_start)

    assert np.allclose(second_fft_torch,second_fft), "Doppler FFT differ"

    save_path2=os.path.join(fft2_folder,f'second_fft_{i}.npy')
    if SAVE_RD:
        np.save(save_path2,second_fft)

    labels_per_sequence=df_labels[df_labels['index'] ==i]
    print('found ',labels_per_sequence.shape[0],' bboxes for index: ',i)
    if labels_per_sequence.shape[0]>0:

        if save_labels:
            
            save_path_labels=os.path.join(label_folder,f'labels_{i}.csv')
            labels_per_sequence.to_csv(save_path_labels,index=False)
    else:
        print(f'no labels found for sequence: {i}')
        ratio_frame+=1
    

    print(i)
    if i>limit_sample:
        break 
    count+=1  

print('Ratio of samples with no bboxes: ',(ratio_frame/count)*100,'%')
sys.exit('exiting  before time benchmark')
#assert len(time_second_fft)==count
print('computing benchmarks....')

# test only to be removed
mean_test=0
for t in time_first_fft:
    mean_test+=t
mean_test=mean_test/len(time_first_fft)

mean_time_v1=np.mean(time_first_fft)
print('Mean time to compute version 1 fft: ',mean_time_v1,' seconds')


assert np.isclose(mean_test,mean_time_v1)

mean_time_v2=np.mean(time_first_fft_v2)
print('Mean time to compute version 2 fft: ',mean_time_v2,' seconds')

mean_time_torch=np.mean(time_first_fft_torch)
print('Mean time to compute fft using torch implementation: ',mean_time_torch,' seconds')

mean_time_second_fft=np.mean(time_second_fft)
print('Mean time to compute second fft (doppler): ',mean_time_second_fft,' seconds')

mean_time_second_fft_torch=np.mean(time_second_fft_torch)
print('Mean time to compute second fft (doppler) using torch: ',mean_time_second_fft_torch)


#### UNCOMMENT IF YOU WANT TO SAVE BENCMARK
# markdown_content = f"""# FFT Timing Comparison Report

# This report includes the mean inference times (in seconds) for various FFT implementations and versions:
# - **Number of samples to compute means**: `{count}`
# - **Mean time to compute FFT (version 1 scipy)**: `{mean_time_v1:.6f}` seconds
# - **Mean time to compute FFT (version 2 matrix DFT + product)**: `{mean_time_v2:.6f}` seconds  
# - **Mean time to compute FFT using PyTorch (torch.fft)**: `{mean_time_torch:.6f}` seconds  
# - **Mean time to compute second FFT (Doppler) scipy**: `{mean_time_second_fft:.6f}` seconds  
# - **Mean time to compute second FFT (Doppler) using PyTorch**: `{mean_time_second_fft_torch:.6f}` seconds  
# """

# # Write to a Markdown file
# with open("fft_versions_report.md", "w") as f:
#     f.write(markdown_content)



print('end of the script')

