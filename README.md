# 3D Segmentation and Multi-label Classification on Congenital Heart Diseases (3D CT-Scan)

## 🔬 Research Preview
This repository contains the official implementation of a novel deep learning pipeline for diagnosing Congenital Heart Disease (CHD) from 3D CT scans.

The methodology is built upon robust data generalization and augmentation feeding a 3D U-Mamba segmentation model, which isolates seven cardiac organs to construct a topology-aware GNN/GCN that analyzes structural deviations for simultaneous, multi-label disease prediction.

## Key Highlights:
1. Frame Generalization: Standardizes highly variable 3D CT volumes to a uniform 256-frame depth utilizing index-based duplication and symmetric trimming.

2. Anatomically-Constrained Augmentation: Enhances dataset diversity through spatial rotation, 3D elastic deformation, and gamma contrast adjustments without compromising medical validity.

3. Topological Modeling (GNN/GCN): Maps seven isolated cardiac nodes against a static adjacency matrix to explicitly evaluate anatomical deviations from a healthy topology.

4. Multi-Label Classification: Utilizes a synthesized Global Cardiac Representation (GCR) vector to perform simultaneous diagnostic predictions across 15 distinct CHD classes.

5. Optimized Loss Strategy: Employs a composite objective function (BCEWithLogitsLoss + focal weighting + asymmetric thresholding) to aggressively heavily penalize the misclassification of rare minority diseases.
