import os
import nibabel as nib
import numpy as np

LABEL_PATH = "/home/s2516118/skripsi/nnUNet_raw/Dataset001_ImageCHD/labelsTr"

for filename in os.listdir(LABEL_PATH):
    if filename.endswith(".nii.gz"):
        path = os.path.join(LABEL_PATH, filename)

        nii = nib.load(path)
        data = nii.get_fdata()

        # 1️⃣ Round float → nearest integer
        data = np.round(data)

        # 2️⃣ Convert to uint8
        data = data.astype(np.uint8)

        # 3️⃣ Remove unexpected labels
        data[data > 7] = 0

        # Save back
        new_nii = nib.Nifti1Image(data, nii.affine, nii.header)
        nib.save(new_nii, path)

        print(f"Cleaned {filename}")

print("✅ All labels cleaned successfully.")

import os
import nibabel as nib
import numpy as np

LABEL_PATH = "/home/s2516118/skripsi/nnUNet_raw/Dataset001_ImageCHD/labelsTr"

all_labels = set()

for filename in os.listdir(LABEL_PATH):
    if filename.endswith(".nii.gz"):
        data = nib.load(os.path.join(LABEL_PATH, filename)).get_fdata()
        unique = np.unique(data)
        all_labels.update(unique)

print("Final unique labels:", sorted(all_labels))
