"""
Fix AllenToTensor multiprocessing issue and extract data exactly as notebook does
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../neuro_ai_course/notebook-2'))

# Monkey-patch the compute_raster_plot to avoid multiprocessing
from allen_to_tensor import AllenToTensor
import numpy as np
from tqdm import tqdm

def compute_raster_plot_fixed(self):
    """Fixed version without multiprocessing"""
    if self.verbose: print("Starting raster plot computation (single-threaded)...")
    session = self.get_session()

    scene_presentations = self.get_scene_presentations_of_stim()
    time_lines = self.get_time_line()

    if self.verbose: print("time line shape: ", time_lines.shape)

    unit_indices = self.get_units_indices()

    spikes = session.presentationwise_spike_times(
        stimulus_presentation_ids=scene_presentations.index.values,
        unit_ids=unit_indices
    )

    start_times = time_lines
    stop_times = time_lines + self.dt

    if self.verbose:
        print("Number of units:", unit_indices.size)
        print("Number of spikes:", len(spikes.index))

    n_units = len(unit_indices)
    spike_times_list = []
    for k_unit in range(n_units):
        unit_idx = unit_indices[k_unit]
        z = spikes[spikes["unit_id"] == unit_idx]
        spike_times = np.array([z.index[k] for k in range(len(z.index))])
        spike_times_list.append(spike_times)

    print(f"Processing {n_units} units (single-threaded)...")

    rasters = []
    for spike_times in tqdm(spike_times_list):
        sub_raster = np.zeros((time_lines.shape[0], time_lines.shape[1],), dtype=np.int8)

        for spike_time in spike_times:
            wh = np.where(np.logical_and(spike_time >= start_times, spike_time < stop_times))

            if len(wh[0]) > 1:
                print("Spike time:", spike_time)
                print("time bin positions:", wh)
                raise ValueError("A spike can be in only one time bin, got {}".format(len(wh[0])))
            elif len(wh[0]) == 0:
                pass
            else:
                sub_raster[wh[0][0], wh[1][0]] = 1

        rasters.append(sub_raster)

    raster = np.stack(rasters, 2)
    assert raster.sum() > 0

    print("The raster of shape {} has {} spikes. (total spikes={})".format(raster.shape, np.sum(raster), len(spikes.index)))
    return raster

# Monkey-patch the method
AllenToTensor.compute_raster_plot = compute_raster_plot_fixed

# Now use AllenDriftingDataset
from allen_dataset import AllenDriftingDataset

print("="*60)
print("Loading AllenDriftingDataset (this will take a while)...")
print("="*60)

cache_root = os.path.join(os.path.dirname(__file__), '../')
dataset = AllenDriftingDataset(
    train_fraction=0.66,
    dt=0.01,
    cache_root=cache_root
)

print("\n" + "="*60)
print("Dataset loaded successfully!")
print("="*60)

# Get raster
print(f"\nRaster list: {len(dataset.raster_list)} sessions")
print(f"Raster shape: {dataset.raster_list[0].shape}")

# Get test split
test_indices = dataset.get_split_specific_trial_selection(0, "test")
print(f"Test indices: {len(test_indices)} trials")

# Extract test data
z_test = dataset.raster_list[0][test_indices]
print(f"z_test shape: {z_test.shape}")
print(f"z_test dtype: {z_test.dtype}")
print(f"Total spikes: {z_test.sum()}")

# Calculate size
size_mb = z_test.nbytes / (1024**2)
print(f"z_test size: {size_mb:.2f} MB")

# Save
output_dir = os.path.join(os.path.dirname(__file__), '../neural_activity/data')
os.makedirs(output_dir, exist_ok=True)

metadata = {
    'n_test': z_test.shape[0],
    'n_time': z_test.shape[1],
    'n_neurons': z_test.shape[2],
    'dt': 0.01,
    'area': 'all',  # AllenDriftingDataset uses all areas by default
    'source': 'allen_drifting_dataset',
    'session_id': 791319847,
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
print(f"\n✓ Data saved to {output_dir}/neural_data.npz")
print(f"File size (compressed): {file_size_mb:.2f} MB")
print(f"Metadata: {metadata}")
