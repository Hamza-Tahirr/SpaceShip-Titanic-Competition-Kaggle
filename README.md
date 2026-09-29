# Spaceship Titanic (Kaggle)

My notebook for the Kaggle getting-started competition [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic). The task is binary classification: predict whether each passenger was transported to an alternate dimension (`Transported`) after the ship hit a spacetime anomaly. The notebook explores the data, adds a few engineered features and trains a LightGBM classifier to produce a submission file.

## What the notebook does

- **Exploration**: summary statistics, count plots of the categorical features (overall and split by `Transported`), histograms and box plots of age and spending, and heatmaps of missing values.
- **Cleaning**: drops `PassengerId` and `Cabin` from the features, fills missing numeric values with the column mean and missing categorical values with the most frequent value.
- **Feature engineering**:
  - ordinal encoding of `HomePlanet`, `CryoSleep`, `Destination` and `VIP`
  - `Age` binned into five groups
  - `Family_size`: the number of passengers who share a surname
  - `TotalAmount`: total spending across the five amenities
  - a 0/1 usage flag for each amenity
- **Modelling**: 80/20 stratified train/validation split, `StandardScaler`, then a 5-fold `GridSearchCV` over LightGBM hyperparameters. A final `LGBMClassifier` is trained with a fixed parameter set and checked with training/validation accuracy and a confusion matrix.
- **Submission**: predicts the test set and writes `submission.csv` in the competition format (`PassengerId`, `Transported`).

## Dataset

The data comes from the [competition page](https://www.kaggle.com/competitions/spaceship-titanic/data) and is not included in this repository. It has three files: `train.csv`, `test.csv` and `sample_submission.csv`.

Columns in `train.csv`: `PassengerId`, `HomePlanet`, `CryoSleep`, `Cabin`, `Destination`, `Age`, `VIP`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`, `Name` and the target `Transported`.

## Observations

The saved notebook only has outputs for the first cells, which load the data and print summary statistics. To see the plots, the grid search results and the accuracy scores, run the notebook again. The saved outputs show that:

- the training set has 8,693 rows and 14 columns
- every column except `PassengerId` and `Transported` has missing values, about 180 to 220 each (roughly 2%)
- passenger ages range from 0 to 79, with a mean of about 29
- spending is very skewed: for all five amenities the median is 0 and the 75th percentile is below 80, while the maximum is above 14,000

## Tech stack

Python, Jupyter, pandas, NumPy, Matplotlib, seaborn, scikit-learn and LightGBM.

## Project structure

```
.
├── getting-started-with-spaceship-titanic.ipynb   # analysis, model and submission
├── requirements.txt
├── LICENSE
└── README.md
```

## How to run

**On Kaggle:** create a notebook in the competition and import this one. It reads the data from `/kaggle/input/spaceship-titanic`.

**Locally:**

1. Download the competition files from the Kaggle data page. If you use the Kaggle CLI, join the competition first, then run `kaggle competitions download -c spaceship-titanic`. Put `train.csv` and `test.csv` in a `data/` folder next to the notebook.
2. Install the dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt jupyter
   ```

3. Open the notebook and run all cells:

   ```bash
   jupyter notebook getting-started-with-spaceship-titanic.ipynb
   ```

The grid search tries 864 parameter combinations with 5-fold cross-validation, so it takes a while. The next cell trains the final model with fixed parameters, so you can skip the grid search cell if you only need the submission. `submission.csv` is saved in the working directory.

## License

This project is under the MIT License. See [LICENSE](LICENSE).
