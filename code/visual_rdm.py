import os
import glob
import numpy as np
import pandas as pd
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.rdm.calc import calc_rdm_unbalanced
import cv2 

from rsatoolbox.vis.rdm_plot import show_rdm
import matplotlib.pyplot as plt

# Lists for subjects, runs, and chunks
chunk_list = [0, 59, 68, 55, 71, 54, 68, 83, 51]
run_list = ["01"]

# Path to the movie
movie_path = "/home/exp-psy/Desktop/study_face_tracks/derivatives/fgav/fg_av_ger_seg1.mkv"

# helper functions
def create_dirs():
    """Create output directories for saving fixation density maps and RDMs."""
    output_dir = os.path.join("/home/exp-psy/Desktop/study_face_tracks", "derivatives", "model_rdms", "visual_properties")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_events(run):
    """
    Load event files and use them as onsets and durations to slice the eye-movement data.
    Converts onsets and durations from seconds to frames for a movie presented at 25 fps.
    """
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
    return list(zip(tmp_df["onset"].astype(int), tmp_df["offset"].astype(int))), tmp_df["trial_type"]

def extract_average_frames(movie_path, frame_ranges):
    """Extract frames from a movie, average them, and vectorize the result."""
    cap = cv2.VideoCapture(movie_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {movie_path}")

    averages = []

    for start_frame, end_frame in frame_ranges:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                break

            # Convert frame to grayscale and store it
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)

        # Compute the average frame
        if frames:
            average_frame = np.mean(frames, axis=0)
            averages.append(average_frame.flatten())

    cap.release()
    return np.array(averages)

# create output
output_dir = create_dirs()

for run in run_list:
    steps_list, event_labels = load_events(run)
    print(f"Processing run-{run}: {len(steps_list)} chunks")

    # Extract average frames for each event
    patterns = extract_average_frames(movie_path, steps_list)

    descs = {"sub": "video", "task": "study_face_tracks"}
    model_dataset = Dataset(
        measurements=patterns,
        descriptors=descs,
        obs_descriptors=dict(
            run=np.repeat(int(run), len(event_labels.values)),
            condition=event_labels.values
        )
    )

    model_rdm = calc_rdm_unbalanced(
        dataset=model_dataset,
        method="crossnobis",
        descriptor="condition",
        # cv_descriptor="run" # wieder anmachen?
    )

    model_rdm.save(os.path.join(output_dir, f"visual-RDM.hdf5"))

    fig, _, _ = show_rdm(model_rdm, show_colorbar='panel')
    plt.show()
