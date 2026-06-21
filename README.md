# 3D Segmentation and Multi-label Classification on Congenital Heart Diseases (3D CT-Scan)

## 🔬 Research Preview
This repository explainig the Thesis reseacrh of a deep learning pipeline for diagnosing Congenital Heart Disease (CHD) from 3D CT scans. The methodology is built upon data generalization and augmentation feeding a 3D U-Mamba segmentation model, which isolates seven cardiac organs to construct a topology-aware GNN/GCN that analyzes structural deviations for simultaneous, multi-label disease prediction.

## Research Methodologies:
1. Frame Generalization: Standardizes highly variable 3D CT volumes to a uniform 256-frame depth utilizing index-based duplication and symmetric trimming.
2. Anatomically-Constrained Augmentation: Enhances dataset diversity through spatial rotation, 3D elastic deformation, and gamma contrast adjustments without compromising medical validity.
3. Topological Modeling (GNN/GCN): Maps seven isolated cardiac nodes against a static adjacency matrix to explicitly evaluate anatomical deviations from a healthy topology.
4. Multi-Label Classification: Utilizes a synthesized Global Cardiac Representation (GCR) vector to perform simultaneous diagnostic predictions across 15 distinct CHD classes.
5. Optimized Loss Strategy: Employs a composite objective function (BCEWithLogitsLoss + focal weighting + asymmetric thresholding) to aggressively heavily penalize the misclassification of rare minority diseases.

## Data Source
The dataset used in this project is from Kaggle:
[https://www.kaggle.com/datasets/xiaoweixumedicalai/imagechd](https://www.kaggle.com/datasets/xiaoweixumedicalai/imagechd).

To extract the dataset files:
1. Download the zipped file from the Kaggle link.
2. You might find the downloaded file has a non-standard extension (e.g., ".change2zip"). **Rename this file's extension to ".zip"** (e.g., `ImageCHD_dataset.change2zip` becomes `ImageCHD_dataset.zip`).
3. **Extract all files** from the renamed `.zip` archive.

## Research Pipeline
### (`/Preprocessing`) : Data Pre-processing 
The files inside this folder will demonstrate the codes on step-by-step procedure :
1. * **`labelcleaner.py`** cleans up 3D medical image files by converting decimals into simple whole numbers, shrinking their file sizes, and deleting any unexpected categories higher than 7, and then it double-checks all the images at the end to guarantee that only the correct labels are left.
2. Preparing the U-Mamba framework requires configuring system directories and deep learning dependencies, followed by executing its built-in preprocessing pipeline shown in the * **`umamba.pbs`** script.
3. * **`Frame_Generalization.ipynb`** then introduced because each patient varies within the raw NIfTI files.
   To ensure consistency for the deep learning models, all patient scans are standardized to a uniform length of 256 frames.
    * **For patients with <256 frames:** Frames are duplicated from the middle sequence where the heart is clearly visible.
        * *Example 1:* For a patient with 206 frames, 69 data points (frames 103-172) are duplicated from the middle to reach a total of 256 frames.
        * *Example 2:* For a patient with 137 frames, the entire set of frames is duplicated and repeated to reach 256 frames.
    * **For patients with >256 frames:** Excess frames are removed to standardize the length. For example, if a patient has 340 frames, 65 frames (340 - 256) are removed: 33 from the beginning and 32 from the end, ensuring an even distribution of removal.

### (`/Segmentation`) : 3D U-Mamba Segmentation
In this research, U-Mamba framework is utilized to perform high-resolution 3D volumetric segmentation, leveraging its advanced global spatial awareness to accurately isolate seven distinct cardiac structures from the complex background anatomy. After running the umamba.pbs, the results would be the folders: (`/nnUNet_raw`), (`/nnUNet_preprocessed`), (`/nnUNet_results`).

### (`/Classification`) : Graph Neural Network and Graph Convolution Network
#### 1. Data Preparation & Loading
* **`prep.py`**: An intelligent dataset splitter designed to handle multi-label data and extreme class imbalance. It utilizes `iterative_train_test_split` to carefully distribute normal hearts, common diseases, and extremely rare anomalies (e.g., PAS and AAH) evenly across the training, validation, and testing sets. **This script generates the final dataset splits and outputs them into the `/info` directory as three distinct files: `train_data.csv`, `val_data.csv`, and `test_data.csv**`.
* **`dataset.py`**: Bridges the raw 3D images and the neural network by defining the `CHDMultiLabelDataset` class. It integrates the **MONAI** framework to apply anatomically-constrained 3D augmentations, dynamically triggering `get_heavy_transforms()` (such as 3D elastic deformation) specifically when a rare disease is detected in the training batch.

#### 2. The Topological Brain
* **`model.py`**: The mathematical core of the classification pipeline.
* **Adjacency Matrix**: Defines `get_cardiac_adjacency_matrix()`, which mathematically hardcodes normal circulatory pathways (e.g., connecting the Left Atrium to the Left Ventricle).
* **Feature Extraction**: Uses the `CHDTopologyClassifier` to extract high-level spatial features from the pre-trained U-Mamba backbone.
* **Organ Isolation**: Performs ROI pooling to isolate the 7 specific structural organs and feeds them as nodes into the `GraphConvolutionLayer` (GCN).
* **Disease Prediction**: Synthesizes a Global Cardiac Representation (GCR) and passes it through a classification head to simultaneously predict 15 distinct Congenital Heart Diseases.

#### 3. Training & Evaluation
* **`train.py`**: Executes the highly optimized learning process. It directly tackles the minority class bottleneck by utilizing a `WeightedRandomSampler` (forcing the model to sample rare diseases more frequently) and a custom `MultiLabelFocalLoss` function, which aggressively penalizes the network for misclassifying minority classes.
* **`evaluate.py`**: The final diagnostic evaluator. It loads the trained model weights, processes the test dataset, and applies custom `class_thresholds` (e.g., requiring 50% confidence for a common ASD, but only 20% confidence for a rare PAS) to calculate the final evaluation metrics, including F1, ROC-AUC, and the overall mAP scores.
* **`train.pbs`**: The job submission script for the JAIST Kagayaki HPC, allocating 16 CPU cores, 256GB of RAM, and a high-end GPU for 72 hours to manage the massive computational load of the multi-label training pipeline.

### Additional Information
* **`ImageCHD_Info.ipynb`**: A custom exploratory notebook utilized to analyze and summarize the raw ImageCHD dataset. It actively catalogs 110 NIfTI files and extracts critical metadata, including the Patient ID, filename, and exact spatial dimensions (X, Y, Z)—specifically mapping the Z-axis to determine the total number of volumetric frames for each individual scan.
* **`3d.ipynb`**: Interactive visualization notebook designed to load the processed NIfTI data and generate rotatable 3D models of the cardiac structures. This aids in the visual and spatial validation of the dataset prior to training.




