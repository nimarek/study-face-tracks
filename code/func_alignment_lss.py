import os
import sys
import json
import pandas as pd
import numpy as np
import nibabel as nib
from nltools.data import Brain_Data, Design_Matrix
from nltools.file_reader import onsets_to_dm
from nltools.stats import align, zscore

# I/O
sub = str(sys.argv[1])
all_data = []
target_run, train_run = 1, 7
sub_list = ["01", "02", "03", "04", "06", "10", "14", "15", "16", "17", "18", "19", "20"]
event_file = "/home/exp-psy/Desktop/study_face_tracks/derivatives/reference_face-tracks/studyf_run-01_face-orientation.csv"
deriv_dir = "/home/exp-psy/Desktop/study_face_tracks/derivatives/"
lut_df = pd.read_csv("/home/exp-psy/Desktop/study_face_tracks/derivatives/fmriprep_mni/desc-aparcaseg_dseg.tsv", sep="\t")

# select columes to slice bold data
if target_run == 1:
    train_volumes = 451
elif target_run == 2:
    train_volumes = 441
elif target_run == 3:
    train_volumes = 438
elif target_run == 4:
    train_volumes = 488
elif target_run == 5:
    train_volumes = 462
elif target_run == 6:
    train_volumes = 439
elif target_run == 7:
    train_volumes = 542
elif target_run == 8:
    train_volumes = 338

# nuis. regressors to keep
conf_keep_list = [
    "trans_x", "trans_y", "trans_z", 
    "rot_x", "rot_y", "rot_z"
]

# additional nuis. regressors
conf_add_list = [
    "csf", "white_matter", 
    "a_comp_cor_00", "a_comp_cor_01", 
    "a_comp_cor_02", "a_comp_cor_03", 
    "a_comp_cor_04"
]

# create helper functions
def slicer(in_path, train_volumes):
    img = nib.load(in_path)
    img_data = img.get_fdata()
    sliced_data = img_data[..., :train_volumes]
    print("shape of the training data:\t", sliced_data.shape)
    return nib.Nifti1Image(sliced_data, affine=img.affine, header=img.header)

def make_motion_covariates(mc, tr):
    z_mc = zscore(mc)
    all_mc = pd.concat([z_mc, z_mc**2, z_mc.diff(), z_mc.diff()**2], axis=1)
    all_mc.fillna(value=0, inplace=True)
    return Design_Matrix(all_mc, sampling_freq=1/tr)

def reorder_columns(df):
    priority = ["right-", "left-", "frontal-"]
    prioritized = [col for col in df.columns if any(col.startswith(prefix) for prefix in priority)]
    others = [col for col in df.columns if col not in prioritized]
    reordered_columns = prioritized + others
    return df[reordered_columns]

# define lists of labels to keep
cortical_rois = [f"ctx-{h}-{region}" for h in ["lh", "rh"] for region in [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
    "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
    "insula", "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal",
    "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal",
    "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis",
    "pericalcarine", "postcentral", "posteriorcingulate", "precentral",
    "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal",
    "superiorfrontal", "superiorparietal", "superiortemporal",
    "supramarginal", "frontalpole", "temporalpole", "transversetemporal"
]]

subcortical_rois = [
    "Left-Thalamus-Proper", "Right-Thalamus-Proper",
    "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen",
    "Left-Pallidum", "Right-Pallidum",
    "Left-Hippocampus", "Right-Hippocampus",
    "Left-Amygdala", "Right-Amygdala",
    "Left-Accumbens-area", "Right-Accumbens-area"
]

# combine cortical and subcortical ROIs
rois_of_interest = cortical_rois + subcortical_rois
filtered_data = lut_df[lut_df["name"].isin(rois_of_interest)]

for roi in filtered_data["name"]:
    print(f"working on roi: {roi}")
    matches = lut_df[lut_df["name"] == roi]
    match_index = matches["index"].values[0]

    # create output folder
    out_dir = os.path.join(deriv_dir, "hyperalignment", f"sub-{sub}", f"roi-{roi}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    else:
        print(f"path exists: {out_dir}")
        continue

    # start loading data
    for sub_train in sub_list:
        aparc_fpath = os.path.join(
            "/home", 
            "exp-psy", 
            "Desktop", 
            "study_face_tracks", 
            "derivatives", 
            "fmriprep_mni",
            f"sub-{sub_train}", 
            "ses-movie", 
            "func", 
            f"sub-{sub_train}_ses-movie_task-movie_run-{train_run}_space-MNI152NLin2009cAsym_res-2_desc-aparcaseg_dseg.nii.gz"
        )
        affine = nib.load(aparc_fpath).affine 
        aparc_data = nib.load(aparc_fpath).get_fdata()
        
        # create roi from surface reconstruction
        target_mask = np.zeros_like(aparc_data, dtype=bool)
        target_mask[aparc_data == float(match_index)] = True
        roi_mask = nib.nifti1.Nifti1Image(target_mask*1.0, affine=affine)
            
        func_f = os.path.join(
            "/home", 
            "exp-psy", 
            "Desktop", 
            "study_face_tracks", 
            "derivatives", 
            "fmriprep_mni",
            f"sub-{sub_train}", 
            "ses-movie", 
            "func", 
            f"sub-{sub_train}_ses-movie_task-movie_run-{train_run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
            )
        data = Brain_Data(slicer(func_f, train_volumes), mask=roi_mask)
        all_data.append(data)

    # start hyperalignment
    hyperalign = align(all_data, method="procrustes")

    # start least-square separate estimation
    print(f"extracting data from subject: {sub}")
    aparc_fpath = os.path.join(
    "/home", 
    "exp-psy", 
    "Desktop", 
    "study_face_tracks", 
    "derivatives", 
    "fmriprep_mni",
    f"sub-{sub}", 
    "ses-movie", 
    "func", 
    f"sub-{sub}_ses-movie_task-movie_run-{target_run}_space-MNI152NLin2009cAsym_res-2_desc-aparcaseg_dseg.nii.gz"
    )

    func_f = os.path.join(
    "/home", 
    "exp-psy", 
    "Desktop", 
    "study_face_tracks", 
    "derivatives", 
    "fmriprep_mni",
    f"sub-{sub}", 
    "ses-movie", 
    "func", 
    f"sub-{sub}_ses-movie_task-movie_run-{target_run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    )

    affine = nib.load(aparc_fpath).affine 
    aparc_data = nib.load(aparc_fpath).get_fdata()

    # create roi from surface reconstruction
    target_mask = np.zeros_like(aparc_data, dtype=bool)
    target_mask[aparc_data == float(match_index)] = True
    roi_mask = nib.nifti1.Nifti1Image(target_mask*1.0, affine=affine)

    # load target data
    target_data = Brain_Data(func_f, mask=roi_mask)
    print("volumes available:\t", len(target_data))

    # align sub. to common space
    aligned_sub_hyperalignment = target_data.align(hyperalign["common_model"], method="procrustes")

    # get events, confounds, hyperparameters ...
    events = pd.read_csv(event_file, sep=",")
    events = events.rename(
        columns={"onset": "Onset", "duration": "Duration", "trial_type": "Stim"}
    )
    events = events[["Onset", "Duration", "Stim"]]
    events = events[events["Stim"].str.count("frontal|right|left") == 1]

    conf_file = f"/home/exp-psy/Desktop/study_face_tracks/derivatives/fmriprep_mni/sub-{sub}/ses-movie/func/sub-{sub}_ses-movie_task-movie_run-{target_run}_desc-confounds_timeseries.tsv"
    confounds = pd.read_csv(conf_file, sep="\t")[conf_keep_list]

    json_file = f"/home/exp-psy/Desktop/study_face_tracks/derivatives/fmriprep_mni/sub-{sub}/ses-movie/func/sub-{sub}_ses-movie_task-movie_run-{target_run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json"
    with open(json_file, "r") as f:
        metadata = json.load(f)
    TR = metadata.get("RepetitionTime", None)
    print(f"TR: {TR}")

    cov_dm = make_motion_covariates(confounds, TR)
    confounds_add = pd.read_csv(conf_file, sep="\t")[conf_add_list]
    cov_add_dm = Design_Matrix(confounds_add, sampling_freq=1/TR)

    for i, row in events.iterrows():
        lss_df = events.copy()
        trial_type = row["Stim"]

        out_fname_t = os.path.join(out_dir, f"sub-{sub}_run-{target_run}_contrast-{trial_type}_t-map.nii.gz")        
        out_fname_beta = os.path.join(out_dir, f"sub-{sub}_run-{target_run}_contrast-{trial_type}_beta-map.nii.gz")

        print("working on stimulus:\t", trial_type)
        
        lss_df["Stim"] = lss_df["Stim"].apply(lambda x: x if x == row["Stim"] else "other")
        dm = onsets_to_dm(lss_df, 1/TR, train_volumes)
        dm_conv = dm.convolve()
        dm_conv_filt = dm_conv.add_dct_basis(duration=128)
        dm_conv_filt_poly = dm_conv_filt.add_poly(order=2, include_lower=True)
        dm_conv_filt_poly_cov = pd.concat([dm_conv_filt_poly, cov_dm, cov_add_dm], axis=1)
        dm_conv_filt_poly_cov_ordered = reorder_columns(dm_conv_filt_poly_cov)

        # diagnosis
        # print(dm_conv_filt_poly_cov_ordered.columns)
        # dm_conv_filt_poly_cov_ordered.heatmap(cmap="RdBu_r", vmin=-1,vmax=1)
        
        smoothed = aligned_sub_hyperalignment["transformed"].smooth(fwhm=3)
        smoothed.X = dm_conv_filt_poly_cov
        stats = smoothed.regress()
        stats["t"][0].write(out_fname_t)
        stats["beta"][0].write(out_fname_beta)