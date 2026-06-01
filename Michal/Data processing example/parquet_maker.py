import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# =================================================================================== Config
DATASET_ROOT = r"D:\AGH\DS\PP\datasets\ds004504-download" 

TARGET_FS = None  # Target sampling frequency (Hz)
CHANNELS_TO_KEEP = ['Fp1', 'Fp2', 'P3', 'P4']
OUTPUT_FILE = "eeg_preprocessed_4ch_raw.parquet"
SECONDS_TO_PLOT = 5.0 # seconds of signal to plot during snity check

def process_dataset():
    # 1. Load patient information
    participants_path = os.path.join(DATASET_ROOT, 'participants.tsv')
    participants_df = pd.read_csv(participants_path, sep='\t')
    
    processed_data = []

    print(f"Found {len(participants_df)} subjects. Starting processing...")

    for index, row in participants_df.iterrows():
        subject_id = row['participant_id']
        group = row.get('Group', 'Unknown') 
        
        file_path = os.path.join(DATASET_ROOT, 'derivatives', subject_id, 'eeg', f'{subject_id}_task-eyesclosed_eeg.set')
        
        if not os.path.exists(file_path):
            file_path = os.path.join(DATASET_ROOT, subject_id, 'eeg', f'{subject_id}_task-eyesclosed_eeg.set')
            
        if not os.path.exists(file_path):
            print(f"SKIPPING {subject_id} - file not found: {file_path}")
            continue

        print(f"Processing {subject_id} ({group})...")
        
        try:
            # Load data
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            
            # Select electrodes
            raw.pick_channels(CHANNELS_TO_KEEP)
            
            # Resampling
            current_fs = raw.info['sfreq']
            
            if TARGET_FS is not None:
                if current_fs != TARGET_FS:
                    print(f"   Changing Fs: {current_fs}Hz -> {TARGET_FS}Hz")
                    raw.resample(TARGET_FS, npad="auto")
            else:
                print(f"   Kept original Fs: {current_fs}Hz")
            
            # Extract signal matrix
            patient_record = {
                'subject_id': subject_id,
                'group': group,
            }
            
            for ch_name in raw.ch_names:
                signal = raw.get_data(picks=ch_name)[0] 
                patient_record[ch_name] = signal.tolist() 
                
            processed_data.append(patient_record)

            # ======================================================================== Visualisation for the first subject (sanity check)
            if len(processed_data) == 1:
                print(f"\n>>> Generating plot for {subject_id}...")
                print(">>> WARNING: Script is paused. Close the plot window to continue processing!\n")
                
                # Create 4 stacked subplots
                fig, axes = plt.subplots(len(CHANNELS_TO_KEEP), 1, figsize=(10, 8), sharex=True)
                fig.suptitle(f'EEG Signal - {subject_id} ({group}) - First {SECONDS_TO_PLOT} s', fontsize=14)
                
                # Calculate how many samples we need for e.g., 5 seconds
                samples_to_plot = int(SECONDS_TO_PLOT * current_fs)
                time_axis = np.arange(samples_to_plot) / current_fs
                
                for i, ch_name in enumerate(CHANNELS_TO_KEEP):
                    # Take the truncated signal fragment
                    sig_fragment = patient_record[ch_name][:samples_to_plot]
                    
                    axes[i].plot(time_axis, sig_fragment, color='#1f77b4', linewidth=1)
                    axes[i].set_ylabel(ch_name, fontsize=12, fontweight='bold')
                    axes[i].grid(True, linestyle='--', alpha=0.7)
                
                axes[-1].set_xlabel('Time [seconds]', fontsize=12)
                plt.tight_layout()
                plt.show() 

        except Exception as e:
            print(f"ERROR processing {subject_id}: {e}")


    # ================================================================================ Save to parquet
    if not processed_data:
        print("Failed to process any data. Parquet file will not be created.")
        return

    print("\nCreating DataFrame and saving to Parquet file...")
    final_df = pd.DataFrame(processed_data)
    final_df.to_parquet(OUTPUT_FILE, engine='pyarrow')
    print(f"Finished successfully! File size on disk: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    process_dataset()