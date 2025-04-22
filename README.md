# DINO ResNet50 vs. ViT-S/8 on Multi-Label Land Use Classification

This project compares two self-supervised learning models—**ResNet50** and **ViT-S/8**, both pretrained using **DINO**—on a **multi-label land use classification task**. The dataset consists of images labeled with multiple land use tags, and performance is evaluated using metrics like accuracy, precision, recall, and F1 score.

------------------------------------------------------------------------

## Project Structure

-   `dino_resnet50_vs_vits8.ipynb`: Main Jupyter notebook where the full pipeline (data loading, preprocessing, training, evaluation) is implemented.
-   `LandUse_Multilabeled.txt`: Tab-separated label file with image names and multi-label annotations.
-   `Images/`: Directory containing the land use images (assumed structure based on typical use).

------------------------------------------------------------------------

## Setup & Installation

```bash
git clone https://github.com/HingedGuide/DeepLearningMGI12
cd DeepLearningMGI12

# Install dependencies
pip install -r requirements.txt
```
---

## How to Run

### Download Images

Images can be downloaded from the [UCM Data Repository](https://git.wur.nl/lobry001/ucmdata).

- Download the dataset archive from the link above.
- Unzip the contents into an `Images/` folder in the root directory of this project.


1. Ensure your dataset is correctly placed:
   - `LandUse_Multilabeled.txt`
   - `Images/` folder in the expected structure.

2. Launch the notebook:
   ```bash
   jupyter notebook dino_resnet50_vs_vits8.ipynb
   ```

3. Follow the cells step-by-step to run training and evaluation.


------------------------------------------------------------------------

## Evaluation Metrics

The notebook evaluates models using:

-   Accuracy
-   Precision
-   Recall
-   F1 Score

Plots are generated to visualize performance and label-wise statistics.

------------------------------------------------------------------------

## Models Compared

-   **DINOv2 ResNet50**
-   **DINOv2 ViT-S/8**

Both models use feature extraction from the DINOv2-pretrained weights followed by custom classification heads.

------------------------------------------------------------------------

## Notes

-   You can customize transforms, model layers, and hyperparameters in the notebook.
-   GPU is recommended for training.

------------------------------------------------------------------------

## License

MIT License — see `LICENSE` for details.
