import os
import glob
import itertools
import numpy as np
import pandas as pd
from sklearn.utils import resample
from scipy.ndimage import gaussian_filter
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.rdm.calc import calc_rdm_unbalanced
from rsatoolbox.rdm.rdms import load_rdm, concat

from rsatoolbox.vis.rdm_plot import show_rdm
import matplotlib.pyplot as plt

# housekeeping
sigma = 0
resample_rate = 250
eye_raw_dir = os.path.join("/home/exp-psy/Desktop/study_face_tracks", "derivatives", "fix_maps")
rdm_list = []

# Lists for subjects, runs, and chunks
chunk_list = [0, 59, 68, 55, 71, 54, 68, 83, 51]
sub_list = ["01", "02", "03", "04", "05", "06", "09", "10", "14", "15", "16", "17", "18", "19", "20"]
run_list = ["01", "02"]

# helper functions
def create_dirs(sub):
    """Create output directories for saving fixation density maps and RDMs."""
    output_dir = os.path.join("/home/exp-psy/Desktop/study_face_tracks", "derivatives", "model_rdms", "eye_tracks", f"sub-{sub}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_split(sub, root_dir):
    """
    Load preprocessed eye-tracking data.
    """
    file_path = os.path.join(root_dir, f"sub-{sub}_ses-movie_task-movie.npz")
    with np.load(file_path, allow_pickle=True) as data:
        df_movie_frame = data["sa.movie_frame"]
        df_names = data["fa.name"].astype("str")
        df_samples = data["samples"]

    # Create a dataframe from arrays
    df_all = pd.DataFrame(data=df_samples, columns=df_names)
    df_all["frame"] = df_movie_frame
    return df_all

def load_events(run_list):
    """
    Load event files and use them as onsets and durations to slice the eye-movement data.
    Converts onsets and durations from seconds to frames for a movie presented at 25 fps.
    """
    tmp_list, tmp_events, run_ident = [], [], []
    for run in run_list:
        event_path = f"/home/exp-psy/Desktop/study_face_tracks/derivatives/reference_face-tracks/studyf_run-{run}_face-orientation.csv"
        tmp_df = pd.read_csv(event_path, delimiter=",")

        keywords = ["frontal", "right", "left"]

        # pre-process dataframe
        tmp_df = tmp_df[
            tmp_df["trial_type"].apply(lambda x: sum(x.count(keyword) for keyword in keywords) == 1)
        ]
        tmp_df["trial_type"] = tmp_df["trial_type"].str.replace(r"-\d+$", "", regex=True)

        # Convert onset and duration from seconds to frames
        tmp_df["onset"] = tmp_df["onset"] * 25  
        tmp_df["duration"] = tmp_df["duration"] * 25  
        tmp_df["offset"] = tmp_df["onset"] + tmp_df["duration"]
        tmp_list.append(zip(tmp_df["onset"].astype(int), tmp_df["offset"].astype(int)))
        tmp_events.append(tmp_df["trial_type"])
        run_ident.append(np.repeat(int(run), len(tmp_df["trial_type"])))
    return list(itertools.chain(*tmp_list)), list(itertools.chain(*tmp_events)), list(itertools.chain(*run_ident))

def chunk_data(df_all, b_frame, e_frame):
    """Extract a chunk of eye-tracking data based on frame range."""
    chunked_df = df_all.loc[(df_all["frame"] >= b_frame) & (df_all["frame"] <= e_frame)]
    return chunked_df

def plot_norm_data(chunk_data, sigma=None):
    """
    Generate fixation density maps and save them to disk.
    """
    width, height = 1280, 546
    extent = [0, width, height, 0]
    canvas = np.vstack((chunk_data["x"].to_numpy(), chunk_data["y"].to_numpy())) 

    # Bin into image-like format
    hist, _, _ = np.histogram2d(
        canvas[1, :],
        canvas[0, :],
        bins=(height, width),
        range=[[0, height], [0, width]]
    )

    # Smooth the histogram
    hist = gaussian_filter(hist, sigma=sigma)
    return hist.flatten()

for sub in sub_list:
    output_dir = create_dirs(sub)
    exa_df = load_split(sub, root_dir=eye_raw_dir)
    steps_list, event_labels, run_identifier = load_events(run_list)

    print(f"Processing sub-{sub}: {len(steps_list)} chunks")

    # initialize empty arrays
    spatial_patterns = np.full([len(steps_list), 698880], np.nan)

    for chunk_num, ((start_frame, end_frame), label) in enumerate(zip(steps_list, event_labels), start=1):
        # extract eye-tracking data
        chunked_df = chunk_data(df_all=exa_df, b_frame=start_frame, e_frame=end_frame)

        # gen. heatmap and get the histogram
        spatial_patterns[chunk_num - 1, :] = plot_norm_data(chunked_df, sigma=sigma).squeeze()

    descs = {"sub": sub, "task": "study_face_tracks"}
    spatial_dataset = Dataset(
        measurements=spatial_patterns,
        descriptors=descs,
        obs_descriptors=dict(
            run=run_identifier,
            condition=event_labels
        )
    )

    spatial_rdm = calc_rdm_unbalanced(
        dataset=spatial_dataset,
        method="crossnobis",
        descriptor="condition",
        cv_descriptor="run"
    )
    spatial_rdm.save(os.path.join(output_dir, f"sub-{sub}_eye-RDM.hdf5"))
    # # fig, _, _ = show_rdm(model_rdm, show_colorbar="panel")
    # # plt.show()

# spatial
in_fpaths = glob.glob("/home/exp-psy/Desktop/study_face_tracks/derivatives/model_rdms/eye_tracks/sub*/*_eye-RDM.hdf5")
rdms_list = [load_rdm(file_path) for file_path in in_fpaths]
general_model = concat(rdms_list)

fig, _, _ = show_rdm(general_model, show_colorbar="panel")
plt.show()

general_model.save(os.path.join("/home/exp-psy/Desktop/study_face_tracks/derivatives/model_rdms/eye_tracks", f"general_eye-RDM.hdf5"))