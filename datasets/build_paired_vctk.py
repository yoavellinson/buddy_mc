import os
import shutil
import glob
import numpy as np
import pyroomacoustics as pra
import soundfile as sf

# The 4 speakers you requested
TARGET_SPEAKERS = ["p351", "p360", "p226", "p287"]

def generate_rir(output_path, fs=16000):
    """Generates a randomized synthetic Room Impulse Response."""
    # Random room dimensions (L, W, H) in meters
    room_dim = [np.random.uniform(3, 9), np.random.uniform(3, 9), np.random.uniform(2.5, 4)]
    rt60 = np.random.uniform(0.2, 0.45) # Reverberation time
    e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
    
    # Create the room
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_abs), max_order=max_order)
    room.add_source([2.0, 2.0, 1.5])
    room.add_microphone([2.5, 3.0, 1.5])
    room.compute_rir()
    
    # Extract RIR (1D array)
    rir = room.rir[0][0]
    sf.write(output_path, rir, fs)

def build_subset_vctk(source_dir, target_base_dir, fs=16000):
    clean_base = os.path.join(target_base_dir, "clean")
    rir_base = os.path.join(target_base_dir, "rir")

    print(f"Filtering for speakers: {TARGET_SPEAKERS}")

    for spk in TARGET_SPEAKERS:
        # Search for the speaker folder in your source (case-insensitive check)
        spk_pattern = os.path.join(source_dir, spk)
        if not os.path.exists(spk_pattern):
            print(f"⚠️ Warning: Speaker folder {spk} not found in {source_dir}. Skipping...")
            continue

        # Create output subdirectories
        os.makedirs(os.path.join(clean_base, spk), exist_ok=True)
        os.makedirs(os.path.join(rir_base, spk), exist_ok=True)

        # Get all wavs for this speaker
        wav_files = glob.glob(os.path.join(spk_pattern, "*.wav"))
        print(f"Processing {len(wav_files)} files for {spk}...")

        for wav_path in wav_files:
            fname = os.path.basename(wav_path)
            
            # 1. Copy Clean File
            shutil.copy(wav_path, os.path.join(clean_base, spk, fname))
            
            # 2. Generate Matching RIR File
            generate_rir(os.path.join(rir_base, spk, fname), fs=fs)

    print("Dataset build complete.")

# --- RUN THE BUILDER ---
# source_vctk: where your folders p226, p287, etc. are currently located
# output_folder: where the 'clean' and 'rir' structure will be built
build_subset_vctk("/dsi/gannot-lab/gannot-lab1/datasets/VCTK/DS_10283_3443/VCTK-Corpus-0.92/wav16k", "/dsi/gannot-lab/gannot-lab1/datasets/VCTK/DS_10283_3443/VCTK-Corpus-0.92/wav16k_rev")