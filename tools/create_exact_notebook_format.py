"""
Create data in exact same format as notebook:
- Shape: (n_trials, 200, n_neurons)
- Binary spike counts per 10ms bin
- Test split with 34% of data
"""
import numpy as np
import os
import sys

# We need to use the ecephys data that was already downloaded
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../neuro_ai_course/notebook-2'))

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

def create_raster_data():
    """Create raster data matching notebook format"""

    print("Loading session from ecephys cache...")
    cache_dir = os.path.join(os.path.dirname(__file__), '../ecephys_cache_dir')
    manifest_path = os.path.join(cache_dir, "manifest.json")

    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    session_id = 791319847
    session = cache.get_session_data(session_id)

    print(f"Session loaded: {session_id}")

    # Get drifting gratings stimulus
    stim_table = session.get_stimulus_table(stimulus_names="drifting_gratings")
    print(f"Drifting gratings trials: {len(stim_table)}")

    # Get VISp units
    units = session.units
    visp_units = units[units['ecephys_structure_acronym'] == 'VISp']

    # Take first 40 neurons (to keep < 50MB)
    n_neurons = 40
    unit_ids = visp_units.index[:n_neurons].tolist()
    print(f"Using {len(unit_ids)} VISp neurons")

    # Parameters matching notebook
    dt = 0.01  # 10ms bins
    n_time_bins = 200  # 2 seconds of data

    # Collect raster for each trial
    n_trials = min(len(stim_table), 416)  # Match notebook size
    raster = np.zeros((n_trials, n_time_bins, n_neurons), dtype=np.int8)

    print(f"\\nCreating raster of shape {raster.shape}...")

    for trial_idx, (_, trial) in enumerate(stim_table.head(n_trials).iterrows()):
        if trial_idx % 50 == 0:
            print(f"Processing trial {trial_idx}/{n_trials}...")

        start_time = trial['start_time']
        # Use first 2 seconds (200 bins × 10ms)
        trial_duration = n_time_bins * dt
        end_time = start_time + trial_duration

        for neuron_idx, unit_id in enumerate(unit_ids):
            # Get spike times for this unit
            spike_times = session.spike_times[unit_id]

            # Filter spikes in trial window
            trial_spikes = spike_times[
                (spike_times >= start_time) & (spike_times < end_time)
            ]

            # Bin spikes into 10ms bins
            for spike_time in trial_spikes:
                time_bin = int((spike_time - start_time) / dt)
                if 0 <= time_bin < n_time_bins:
                    raster[trial_idx, time_bin, neuron_idx] = 1  # Binary: 1 if spike occurred

    print(f"\\nRaster created!")
    print(f"Shape: {raster.shape}")
    print(f"Total spikes: {raster.sum()}")
    print(f"Mean spike rate: {raster.mean():.4f} spikes/bin")
    print(f"Mean firing rate: {raster.mean() / dt:.2f} Hz")

    # Split into train/test (66/34)
    n_test = int(0.34 * n_trials)
    z_test = raster[-n_test:]

    print(f"\\nTest set: {z_test.shape}")

    # Calculate size
    size_mb = z_test.nbytes / (1024**2)
    print(f"Test set size: {size_mb:.2f} MB")

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '../neural_activity/data')
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        'n_test': z_test.shape[0],
        'n_time': z_test.shape[1],
        'n_neurons': z_test.shape[2],
        'dt': dt,
        'area': 'VISp',
        'source': 'allen_raster',
        'session_id': session_id,
        'stimulus': 'drifting_gratings',
        'format': '(n_trials, n_time, n_neurons)',
        'data_type': 'binary_spikes'
    }

    np.savez_compressed(
        os.path.join(output_dir, 'neural_data.npz'),
        z_test=z_test,
        metadata=metadata
    )

    file_size_mb = os.path.getsize(os.path.join(output_dir, 'neural_data.npz')) / (1024**2)
    print(f"\\n✓ Data saved to {output_dir}/neural_data.npz")
    print(f"File size (compressed): {file_size_mb:.2f} MB")
    print(f"Metadata: {metadata}")

    return z_test, metadata


if __name__ == "__main__":
    try:
        z_test, metadata = create_raster_data()
        print("\\n✓ Data extraction complete!")
    except Exception as e:
        print(f"\\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
