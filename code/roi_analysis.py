import os
import glob
import numpy as np
import pandas as pd
from natsort import natsorted

from nilearn.image import load_img
from rsatoolbox.model.model import ModelFixed
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.inference.evaluate import eval_dual_bootstrap
from rsatoolbox.rdm.calc import calc_rdm_unbalanced
from rsatoolbox.rdm.rdms import concat, load_rdm
from rsatoolbox.vis import plot_model_comparison

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore", UserWarning)

# I/O
sub_list = ["01", "02", "03", "04", "06", "10", "14", "15", "16", "17", "18", "19", "20"]
orientations = ["frontal", "left", "right"]
eye_fpath = f"/home/exp-psy/Desktop/study_face_tracks/derivatives/model_rdms/eye_tracks"
vis_fpath = f"/home/exp-psy/Desktop/study_face_tracks/derivatives/model_rdms/visual_properties"

out_dir_roi = f"/home/exp-psy/Desktop/study_face_tracks/derivatives/roi-results_mni/"
if not os.path.exists(out_dir_roi):
    os.makedirs(out_dir_roi, exist_ok=True)

# housekeeping
rdm_list = []
lut_fpath = "/home/exp-psy/Desktop/study_face_tracks/derivatives/fmriprep_mni/desc-aparcaseg_dseg.tsv"
lut_df = pd.read_csv(lut_fpath, sep="\t")

# create dist. matrix values
def custom_distance(x, y):
    if x == y:
        return 0
    elif (x == "left" and y == "right") or (x == "right" and y == "left"):
        return 0.5
    else:
        return 1

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
rois_of_interest = cortical_rois
filtered_data = lut_df[lut_df["name"].isin(rois_of_interest)]

for roi in filtered_data["name"]:
    print(f"working on roi: {roi}")
    matches = lut_df[lut_df["name"] == roi]
    match_index = matches["index"].values[0]

    for sub in sub_list:
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
            f"sub-{sub}_ses-movie_task-movie_run-1_space-MNI152NLin2009cAsym_res-2_desc-aparcaseg_dseg.nii.gz"
        )
        
        aparc_data = load_img(aparc_fpath).get_fdata()

        # create roi from surface reconstruction
        target_mask = np.zeros_like(aparc_data, dtype=bool)
        target_mask[aparc_data == float(match_index)] = True
        roi_size = target_mask.sum()

        # # create obj. to store data
        # nifti_fpaths = natsorted(
        #     glob.glob(
        #         os.path.join(
        #             "/home", 
        #             "exp-psy", 
        #             "Desktop", 
        #             "study_face_tracks", 
        #             "derivatives", 
        #             "lss_mni_smooth-2_face-tracks", 
        #             f"sub-{sub}", 
        #             f"sub-{sub}*beta-map.nii.gz")
        #     )
        # )

        nifti_fpaths = natsorted(
            glob.glob(
                os.path.join(
                    "/home", 
                    "exp-psy", 
                    "Desktop", 
                    "study_face_tracks", 
                    "derivatives", 
                    "hyperalignment", 
                    f"sub-{sub}", 
                    f"roi-{roi}", 
                    f"sub-{sub}*beta-map.nii.gz")
            )
        )

        nifti_fpaths = [
            file for file in nifti_fpaths
            if sum(file.count(keyword) for keyword in orientations) == 1
            ]

        patterns = np.full([len(nifti_fpaths), roi_size], np.nan)
        conditions = [path.split("/")[-1].split("_")[2].split("-")[1] for path in nifti_fpaths]
        runs = [path.split("/")[-1].split("_")[1].split("-")[1] for path in nifti_fpaths]

        # create distance matrix
        orientation_matrix = np.zeros((len(orientations), len(orientations)))

        for i, label1 in enumerate(orientations):
            for j, label2 in enumerate(orientations):
                orientation_matrix[i, j] = custom_distance(label1, label2)

        # plot matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(orientation_matrix, xticklabels=orientations, yticklabels=orientations, 
                    cmap="coolwarm", annot=True, cbar_kws={"label": "Distance"})
        plt.title("Reduced Custom Distance Matrix")
        plt.xlabel("Labels")
        plt.ylabel("Labels")
        plt.savefig(os.getcwd() + "/similarity_matrix.png")
        plt.close()

        # get beta files
        for c, beta_fpath in enumerate(nifti_fpaths):
            patterns[c, :] = load_img(beta_fpath).get_fdata()[target_mask].squeeze()

        descs = {"sub": sub, "task": "study_face_tracks"}
        ds = Dataset(
            measurements=patterns,
            descriptors=descs,
            obs_descriptors=dict(
                run=runs,
                condition=conditions
            )
        )

        # calculate crossnobis RDMs from the patterns and precision matrices
        rdm_list.append(
            calc_rdm_unbalanced(
                dataset=ds,
                method="crossnobis",
                descriptor="condition",
                cv_descriptor="run"
            )
        )

    # combine datasets and save it
    data_rdms = concat(rdm_list)
    # data_rdms.save(os.path.join(out_dir_nRDM, f"roi-{roi}"))

    models_in = [orientation_matrix, 
                 load_rdm(os.path.join(eye_fpath, "general_eye-RDM.hdf5")), 
                 # load_rdm(os.path.join(vis_fpath, "visual-RDM.hdf5"))
                 ]
    
    model_names = ["face orientation", "FDM"] # ,  "image similarity"
    models_comp = []

    for model, model_name in zip(models_in, model_names):
        models_comp.append(ModelFixed(model_name, model))

    # call function for evaluation
    results = eval_dual_bootstrap(models_comp, data_rdms)
    print(results)
    np.save(os.path.join(out_dir_roi, f"roi-{roi}.npy"), results)

    # plot model
    plot_model_comparison(results, sort=False)
    plt.savefig(os.path.join(out_dir_roi, f"roi-{roi}.png"), dpi=300)
    plt.close()
    
    # clear list obj
    rdm_list.clear()
