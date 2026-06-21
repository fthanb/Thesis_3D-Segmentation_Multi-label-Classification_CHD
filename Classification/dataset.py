import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import monai.transforms as mt

# MULTI-LABEL 3D MEDICAL DATASET
class CHDMultiLabelDataset(Dataset):
    def __init__(self, csv_filepath, image_dir, transform=None, heavy_transform=None):
        
        self.patient_data = pd.read_csv(csv_filepath)
        self.image_dir = image_dir
        self.transform = transform
        self.heavy_transform = heavy_transform
        
        self.patient_data = self.patient_data.fillna(0) 
        
        # remove 1 class
        if 'AVSD' in self.patient_data.columns:
            self.patient_data = self.patient_data.drop(columns=['AVSD'])
            
        self.label_columns = self.patient_data.columns[1:]
        self.num_classes = len(self.label_columns)

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx):
        row = self.patient_data.iloc[idx]
        patient_id = str(int(row['index']))
        
        image_filename = f"{patient_id}_0000.nii.gz" 
        image_path = os.path.join(self.image_dir, image_filename)
        
        labels = row[self.label_columns].values.astype(float)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        
        data_dict = {"image": image_path, "label": label_tensor}
        
        rare_type = labels[2:].sum() > 0
        if self.heavy_transform and rare_type:
            data_dict = self.heavy_transform(data_dict)
        elif self.transform:
            data_dict = self.transform(data_dict)
            
        return data_dict["image"], data_dict["label"]

# DEFINE MONAI TRANSFORMS
def get_standard_transforms():
    
    return mt.Compose([
        mt.LoadImaged(keys=["image"]),
        mt.EnsureChannelFirstd(keys=["image"]), 
        mt.Resized(keys=["image"], spatial_size=(128, 128, 256), mode="trilinear"),
        mt.ScaleIntensityd(keys=["image"]),

        mt.RandRotated(keys=["image"], prob=0.3, range_x=0.2, range_y=0.2, range_z=0.2, mode="bilinear"),
        mt.RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.05),
        mt.ToTensor()
    ])

def get_heavy_transforms():
    
    return mt.Compose([
        mt.LoadImaged(keys=["image"]),
        mt.EnsureChannelFirstd(keys=["image"]), 
        mt.Resized(keys=["image"], spatial_size=(128, 128, 256), mode="trilinear"),
        mt.ScaleIntensityd(keys=["image"]),

        mt.RandRotated(keys=["image"], prob=0.5, range_x=0.25, range_y=0.25, range_z=0.25, mode="bilinear"),

        mt.Rand3DElasticd(
            keys=["image"],
            prob=0.5,
            sigma_range=(5, 7),
            magnitude_range=(50, 100),
            spatial_size=(128, 128, 256),
            mode="bilinear"
        ),

        mt.RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
        mt.RandGaussianSmoothd(keys=["image"], prob=0.5, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        mt.RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 2.0)),
        mt.ToTensor()
    ])

def get_val_test_transforms():
    return mt.Compose([
        mt.LoadImaged(keys=["image"]),
        mt.EnsureChannelFirstd(keys=["image"]), 
        mt.Resized(keys=["image"], spatial_size=(128, 128, 256), mode="trilinear"),
        mt.ScaleIntensityd(keys=["image"]),
        mt.ToTensor()
    ])