from pathlib import Path
import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 500)

import nibabel as nib
from nilearn.image import resample_img, new_img_like
from templateflow.api import get

import arviz as az

def get_data_paths(suffix: str):
    if suffix in ["mask"]:
        pass
    elif suffix in ["z"]:
        suffix = f"stat-{suffix}_statmap"
    
    for subject in range(1, 17):
        yield Path("data") / (
            f"sub-{subject:02d}_task-faces_feature-taskBased_"
            "taskcontrast-facesGtScrambled_"
            "model-aggregateTaskBasedAcrossRuns_"
            f"contrast-intercept_{suffix}.nii.gz"
        )

template_path = get(template="MNI152NLin2009cAsym", resolution=2, desc="brain", suffix="T1w")
template_image = nib.load(template_path)

template_mask_path = get(template="MNI152NLin2009cAsym", resolution=2, desc="brain", suffix="mask")
template_mask_image = nib.load(template_mask_path)
template_mask_data = np.asanyarray(template_mask_image.dataobj, dtype=bool)

target_affine = template_image.affine
target_affine[:3,:3] *= 2.5
target_affine

mask = np.all(
    np.concatenate(
        [
            np.asanyarray(
                resample_img(
                    nib.Nifti1Image.from_filename(mask_path),
                    target_affine,
                    interpolation="nearest",
                ).dataobj
            ).astype(bool)[:, :, :, np.newaxis]
            for mask_path in get_data_paths("mask")
        ], 
        axis=3,
    ),
    axis=3,
)

x, y, z = np.nonzero(mask)
len(x)

data_array = np.concatenate(
    [
        resample_img(
            nib.Nifti1Image.from_filename(zstat_path),
            target_affine,
        ).get_fdata()[x, y, z, np.newaxis]
        for zstat_path in get_data_paths("z")
    ], 
    axis=1,
)

data_frame = pd.DataFrame(data_array)
data_frame["voxel"] = np.ravel_multi_index((x, y, z), mask.shape)

data_frame = data_frame.melt(id_vars=["voxel"], var_name="subject")

df.to_csv("voxel_data.csv", index=False)