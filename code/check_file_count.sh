#!/bin/bash

log_file="missing_files_log.txt"
> "$log_file"

base_dir="/home/exp-psy/Desktop/study_face_tracks/derivatives/hyperalignment"
sub_list=("01" "02" "03" "04" "06" "10" "14" "15" "16" "17" "18" "19" "20")

ROIS=(
    ctx-lh-bankssts ctx-rh-bankssts
    ctx-lh-caudalanteriorcingulate ctx-rh-caudalanteriorcingulate
    ctx-lh-caudalmiddlefrontal ctx-rh-caudalmiddlefrontal
    ctx-lh-cuneus ctx-rh-cuneus
    ctx-lh-entorhinal ctx-rh-entorhinal
    ctx-lh-fusiform ctx-rh-fusiform
    ctx-lh-inferiorparietal ctx-rh-inferiorparietal
    ctx-lh-inferiortemporal ctx-rh-inferiortemporal
    ctx-lh-insula ctx-rh-insula
    ctx-lh-isthmuscingulate ctx-rh-isthmuscingulate
    ctx-lh-lateraloccipital ctx-rh-lateraloccipital
    ctx-lh-lateralorbitofrontal ctx-rh-lateralorbitofrontal
    ctx-lh-lingual ctx-rh-lingual
    ctx-lh-medialorbitofrontal ctx-rh-medialorbitofrontal
    ctx-lh-middletemporal ctx-rh-middletemporal
    ctx-lh-parahippocampal ctx-rh-parahippocampal
    ctx-lh-paracentral ctx-rh-paracentral
    ctx-lh-parsopercularis ctx-rh-parsopercularis
    ctx-lh-parsorbitalis ctx-rh-parsorbitalis
    ctx-lh-parstriangularis ctx-rh-parstriangularis
    ctx-lh-pericalcarine ctx-rh-pericalcarine
    ctx-lh-postcentral ctx-rh-postcentral
    ctx-lh-posteriorcingulate ctx-rh-posteriorcingulate
    ctx-lh-precentral ctx-rh-precentral
)

for sub in "${sub_list[@]}"; do
    for roi in "${ROIS[@]}"; do
        folder="$base_dir/sub-${sub}/roi-${roi}"
        
        if [ -d "$folder" ]; then
            file_count=$(find "$folder" -type f -name "*t-map*.nii.gz" | wc -l)
            echo "Checking $folder - Found $file_count files"
            
            if [ "$file_count" -ne 116 ]; then
                echo "WARNING: Folder $folder has $file_count files instead of 116."
                echo "$folder" >> "$log_file"
            fi
        else
            echo "WARNING: Folder $folder does not exist."
        fi
    done
done

echo "Check completed. See $log_file for missing file details."
