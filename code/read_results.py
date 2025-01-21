import glob
import numpy as np
from nilearn import datasets, plotting, image

# in_f_list = glob.glob(
#     "/home/exp-psy/Desktop/study_face_tracks/derivatives/roi-results_no-func-align/*.npy"
# )

# for f in in_f_list:
#     print(f"working on: {f.split("/")[-1].split(".")[0]}")
#     print(np.load(f, allow_pickle=True))

# # lh-superiortemporal, lh-transversetemporal, rh-superiortemporal, lh-postcentral, rh-inferiortemporal
# # rh-supramarginal. rh-entorhinal, lh-temporalpole, rh-transversetemporal, lh-supramarginal


# Load the Desikan-Killiany atlas
atlas = datasets.fetch_atlas_desikan_killiany()
atlas_img = atlas["maps"]
labels = atlas["labels"]

# Define regions of interest
regions_of_interest = [
    "lh-superiortemporal", "lh-transversetemporal", "rh-superiortemporal",
    "lh-postcentral", "rh-inferiortemporal", "rh-supramarginal",
    "rh-entorhinal", "lh-temporalpole", "rh-transversetemporal", "lh-supramarginal"
]

# Find the indices of the regions in the atlas
roi_indices = [labels.index(region) for region in regions_of_interest]

# Create a mask with only the selected regions
atlas_data = image.load_img(atlas_img).get_fdata()
masked_data = np.zeros_like(atlas_data)

for idx in roi_indices:
    masked_data[atlas_data == idx] = 1

# Convert the masked data back into a NIfTI image
masked_img = image.new_img_like(atlas_img, masked_data)

# Plot the masked image
plotting.plot_roi(masked_img, display_mode="ortho", colorbar=True)
plotting.show()