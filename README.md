# Biomedical Image Segmentation: Skin Lesion Analysis

This repository contains an image processing pipeline for the automatic segmentation of skin lesions (dermoscopy). 

Developed as part of the engineering curriculum at **Télécom Paris**, this project explores the efficacy of **classical Computer Vision** methods (non-Deep Learning) for computer-aided diagnosis.

## Objective
The goal is to isolate the lesion (mole, melanoma) from healthy skin to assist in calculating clinical metrics (symmetry, border, color).

## Methodology (Pipeline)
The project does not use Convolutional Neural Networks (CNNs), but rather a step-by-step algorithmic approach implemented in `final_code.ipynb` :

### 1. Preprocessing
* **Image Resizing** to normalize inputs.
* **Hair Removal:** Using morphological filters (Blackhat) and inpainting to clean the image without altering the lesion structure.
* **Contrast Enhancement** to better distinguish the region of interest.

### 2. Segmentation
* **K-Means Clustering:** The image is converted to a suitable color space (e.g., LAB or HSV), and pixels are grouped into $K$ clusters (Lesion vs. Skin) based on colorimetry.
* **Thresholding:** Binarization of the clustered image.

### 3. Post-processing
* **Mathematical Morphology:** Opening and closing operations to remove noise and fill holes in the predicted mask.
* **Largest Connected Component Selection:** To retain only the main lesion and discard background artifacts.

## Repository Structure
* The most important folder is `final_report`. It contains :  
* `final_code.ipynb` : Jupyter Notebook containing the full code (loading, pipeline, visualization).
* `final_code.pdf` : PDF version of the code and graphical outputs.
* `research_paper.pdf` : Technical report / research paper detailing the algorithm and theoretical results.

## Technologies
* **Python 3**
* **OpenCV (`cv2`)** : Image processing and morphology.
* **Scikit-learn** : K-Means algorithm.
* **NumPy / Matplotlib** : Matrix manipulation and plotting.

## Results
The method is evaluated by comparing the generated mask against the Ground Truth. The results are visible in `final_code.ipynb`, `final_code.pdf`, `research_paper.pdf` and in the folder `results` in the folder `final_report`.
* **Metrics :** Dice Score

---
*Author: Théophile Nadiedjoa (@nadiedjoa-24) and Agshay Nadanakumar (@agshayn) - Télécom Paris*