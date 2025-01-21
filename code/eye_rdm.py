import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sigma = 30
eye_raw_dir = os.path.join("/home/nima/Desktop/diss_proj/study_face_tracks", "derivatives", "fix_maps")

# Lists for subjects, runs, and chunks
chunk_list = [0, 59, 68, 55, 71, 54, 68, 83, 51]
sub_list = ["01", "02", "03", "04", "05", "06", "09", "10", "14", "15", "16", "17", "18", "19", "20"]
# run_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
sub_list= ["01"]
run_list = ["01"]

# Helper Functions
def create_dirs(sub, sigma=None):
    """Create output directories for saving fixation density maps and RDMs."""
    output_dir = os.path.join("/home/nima/Desktop/diss_proj/study_face_tracks", "derivatives", "model_rdms", "eye_tracks", f"sub-{sub}")
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

def load_events(run):
    """
    Load event files and use them as onsets and durations to slice the eye-movement data.
    Converts onsets and durations from seconds to frames for a movie presented at 25 fps.
    """
    event_path = f"/home/nima/Desktop/diss_proj/study_face_tracks/derivatives/reference_face-tracks/studyf_run-{run}_face-orientation.csv"
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
    return list(zip(tmp_df["onset"].astype(int), tmp_df["offset"].astype(int))), tmp_df["trial_type"]

def chunk_data(df_all, b_frame, e_frame):
    """Extract a chunk of eye-tracking data based on frame range."""
    chunked_df = df_all.loc[(df_all["frame"] >= b_frame) & (df_all["frame"] <= e_frame)]
    return chunked_df

def plot_norm_data(sub, run, chunk, chunk_data, sigma=None, output_path="/home/nima/Desktop/diss_proj/study_face_tracks"):
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

    # Plot heatmap
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(
        hist,
        aspect="equal",
        cmap="Blues",
        origin="upper",
        alpha=1,
        extent=extent
    )
    
    save_path = os.path.join(output_path, f"sub-{sub}_run-{run}_scene-{chunk}.png")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return hist

def flatten_arr(eye_arr):
    """
    Compute the Representational Dissimilarity Matrix (RDM) using squared Euclidean distance.
    """
    return [h.flatten() for h in eye_arr]

def save_rdm_csv(rdm, sub, run, output_path):
    """
    Save the RDM as a CSV file.
    """
    csv_path = os.path.join(output_path, f"sub-{sub}_run-{run}_rdm.csv")
    pd.DataFrame(rdm).to_csv(csv_path, header=None, index=False)
    return csv_path

def plot_rdm(rdm, sub, run, output_path):
    """
    Plot the RDM and save as an image.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Remove matrix labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Plot the RDM
    cax = ax.imshow(rdm, cmap="RdBu_r", interpolation="nearest")
    
    # Make the colorbar as big as the matrix plot
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(cax, cax=cbar_ax)
    
    # Remove colorbar labels
    cbar.ax.tick_params(labelsize=0, length=0)
    
    # Save the figure
    plt.savefig(os.path.join(output_path, f"sub-{sub}_run-{run}_rdm.png"), bbox_inches="tight")
    plt.close()

for sub in sub_list:
    output_dir = create_dirs(sub, sigma=sigma)
    exa_df = load_split(sub, root_dir=eye_raw_dir)

    for run in run_list:
        steps_list, event_labels = load_events(run)
        print(f"Processing sub-{sub} run-{run}: {len(steps_list)} chunks")

        # Initialize an empty DataFrame to store the results
        results_df = pd.DataFrame(columns=["trial_type", "eye_arr"])

        for chunk_num, ((start_frame, end_frame), label) in enumerate(zip(steps_list, event_labels), start=1):
            print(f"Chunk-{chunk_num}; Start frame: {start_frame}, End frame: {end_frame}")
            
            # Extract chunked eye-tracking data
            chunked_df = chunk_data(df_all=exa_df, b_frame=start_frame, e_frame=end_frame)
            
            # Generate the heatmap and get the histogram
            hist = flatten_arr(plot_norm_data(sub, run, chunk_num, chunked_df, sigma=sigma, output_path=output_dir))
            
            # Append the trial type and histogram to the results DataFrame
            results_df = pd.concat(
                [results_df, pd.DataFrame({"trial_type": [label], "eye_arr": [hist]})],
                ignore_index=True
            )

        # Save the DataFrame as a CSV file for later use
        csv_path = os.path.join(output_dir, f"sub-{sub}_run-{run}_results.csv")
        results_df.to_csv(csv_path, index=False)

        print(f"Results saved to {csv_path}")


        # # Compute RDM
        # rdm = flatten_arr(eye_arr)
        
        # # Save RDM
        # rdm_csv_path = save_rdm_csv(rdm, sub, run, output_path=output_dir)
        # print(f"Saved RDM as CSV: {rdm_csv_path}")

        # # Plot and save RDM
        # plot_rdm(rdm, sub, run, output_path=output_dir)
        # print(f"Saved RDM plot for sub-{sub} run-{run}")
