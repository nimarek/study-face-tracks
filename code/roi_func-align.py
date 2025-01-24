import os
import sys

from nltools.data import Brain_Data
from nltools.mask import expand_mask
from nltools.stats import align

# I/O
run = str(sys.argv[1])
sub_list = ["01", "02", "03", "04", "06", "10", "14", "15", "16", "17", "18", "19", "20"]
roi = 1
deriv_dir = "/home/exp-psy/Desktop/study_face_tracks/derivatives/"
out_dir = os.path.join(deriv_dir, "hyperalignmend")
print("output folder:\t", out_dir)

# store output
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

all_data = []
for sub in sub_list:
    # paths
    aparc_fpath = os.path.join(
        "/home", 
        "exp-psy", 
        "Desktop", 
        "study_face_tracks", 
        "derivatives", 
        "fmriprep_native",
        f"sub-{sub}", 
        "ses-movie", 
        "func", 
        f"sub-{sub}_ses-movie_task-movie_run-{run}_space-T1w_desc-aparcaseg_dseg.nii.gz"
    )
    print(f"extracting data from subject: {sub} for roi: {roi}")

    # load mask
    mask_data = Brain_Data(aparc_fpath)
    mask_rois = expand_mask(mask_data)
    roi_mask = mask_rois[roi]

    func_f = os.path.join(
        "/home", 
        "exp-psy", 
        "Desktop", 
        "study_face_tracks", 
        "derivatives", 
        "fmriprep_native",
        f"sub-{sub}", 
        "ses-movie", 
        "func", 
        f"sub-{sub}_ses-movie_task-movie_run-{run}_space-T1w_desc-preproc_bold.nii.gz"
        )
    data = Brain_Data(func_f)
    all_data.append(data.apply_mask(roi_mask))

"""
finally start hyperalignment
"""

hyperalign = align(all_data[:15], method='procrustes')