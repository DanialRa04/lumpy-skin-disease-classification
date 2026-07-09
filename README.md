# Lumpy Skin Disease Classification

I used this project to study how environmental and geographic features relate to lumpy skin disease reports. The notebook compares KNN, Random Forest, and a linear SVM after removing metadata columns that would leak the label through their missing-value pattern.

## Dataset

- File: `data/Lumpy_skin_disease_data.csv`
- Rows: 24,803
- Target: `lumpy` where `1` means a reported case and `0` means no reported case

The original file also includes `region`, `country`, and `reportingDate`, but I dropped them during modeling because they are missing for every class-0 row. Keeping them would turn missingness into a shortcut for the target.

## Folder structure

- `codes/SkinDiseaseClassifier.ipynb`: cleaned and executed notebook with saved outputs
- `data/Lumpy_skin_disease_data.csv`: source dataset used by the notebook

## Method

I kept the original class imbalance and scored models with positive-class F1 and balanced accuracy instead of relying on plain accuracy. Hyperparameters are tuned only on the training split with stratified 5-fold cross-validation, and the notebook keeps one held-out test split for the final comparison.

## Setup

Install the usual notebook stack before running:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Then open the notebook from the project root or from the `codes` folder so the relative path to `../data/Lumpy_skin_disease_data.csv` still works.

## Results

In my saved run, Random Forest was the strongest model with test F1 close to `0.89`. KNN was still competitive, and the linear SVM reached high recall but much lower precision.

## Limits

This dataset is imbalanced, so accuracy can look better than the real minority-class performance. The notebook also stays with tabular metadata only, which means it cannot use image-level evidence or outbreak timelines beyond the columns already provided.
