# Credit Card Default Prediction

A data mining project investigating whether customer demographics, previous repayment
behaviour, credit limits, and billing/payment information can help predict next-month
credit-card default.

## Research question

Can a client's previous repayment history, credit limit, and billing information help
predict whether they will default on their credit-card payment next month?

## Dataset

The project uses the **Default of Credit Card Clients** dataset. The dataset contains
30,000 customer records and the target column `default.payment.next.month`
(`0` = no default, `1` = default).

The dataset is not included in this repository. Download the CSV and place it at:

```text
data/UCI_Credit_Card.csv
```

The notebook was originally developed in Google Colab. Its data-loading cell has been
adapted to use this repository-relative path.

Dataset sources:
- [Kaggle dataset copy](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- [UCI Machine Learning Repository record](https://doi.org/10.24432/C55S3H)

## Project workflow

1. **Exploratory data analysis:** target distribution, customer age, credit limits,
   repayment history, billing/payment variables, and default rates.
2. **Data quality checks:** missing values and duplicate records (including duplicate
   checks after excluding the ID column).
3. **Statistical analysis:** numerical correlations, a chi-square test of education
   category versus default, Cramér's V, and an age-stratified descriptive comparison.
4. **Preprocessing:** an 80:20 stratified train/test split; `StandardScaler` for
   numerical variables; `OneHotEncoder` for categorical variables; preprocessing
   fitted within Scikit-learn pipelines to avoid test-set leakage.
5. **Classification:** Logistic Regression and Random Forest, compared with a
   majority-class baseline.
6. **Decision threshold analysis:** precision, recall, F1, false positives, and false
   negatives compared at thresholds 0.3, 0.4, 0.5, and 0.6.
7. **Fairness analysis:** model performance compared across sex and age groups.
8. **Mixed-data similarity:** a worked Gower-distance example using numerical and
   categorical customer attributes.

## Selected results

The dataset's default rate is approximately **22.12%**.

At the default 0.5 threshold, Logistic Regression achieved:

| Metric | Result |
|---|---:|
| Accuracy | 82.10% |
| Precision | 68.47% |
| Recall | 35.37% |
| F1-score | 46.64% |
| ROC-AUC | 0.7614 |

Random Forest achieved 81.78% accuracy and 0.7707 ROC-AUC. The majority-class baseline
achieved 77.87% accuracy but zero precision, recall, and F1 for the default class.

A 0.3 Logistic Regression threshold was selected as a working choice to prioritise
identifying more potential defaulters. At this threshold, recall was 47.21%, precision
was 56.91%, and F1-score was 51.61%. It produced 474 false positives and 700 false
negatives. This is a trade-off, not a proven financially optimal threshold; the actual
costs of the two error types were not available.

The fairness analysis found similar recall for male and female customers (47.63% and
46.90%), while false-positive rates were 12.31% and 8.85%, respectively. Recall varied
across age groups. The oldest test group contained only 67 customers, so its estimates
should be interpreted cautiously.

The worked Gower-distance example produced a distance of **0.3564** between two
customers across six selected attributes.

## Repository structure

```text
credit-card-default-prediction/
├── data/
│   └── README.md
├── notebooks/
│   └── credit_card_default_analysis.ipynb
├── reports/
├── .gitignore
├── README.md
└── requirements.txt
```

## How to run

1. Clone or download this repository.
2. Download `UCI_Credit_Card.csv` from the dataset source and place it in `data/`.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open `notebooks/credit_card_default_analysis.ipynb` in Jupyter Notebook, JupyterLab,
   or Google Colab.
5. Run the notebook cells from top to bottom.

## Limitations

- The models do not identify all actual defaulters; recall remains limited.
- The 0.3 threshold is a chosen operating point, not a cost-optimised threshold.
- Group-level fairness comparisons describe differences in this test sample and do
  not, by themselves, establish discrimination or causation.
- Some demographic category codes in the dataset are undocumented.
- The Gower distance is demonstrated on two customers and six selected attributes;
  it is not a full pairwise customer-similarity analysis.
- The project was evaluated on the available dataset and was not deployed as a
  production lending system.

## References

Yeh, I-C., & Lien, C-H. (2009). The comparisons of data mining techniques for the
predictive accuracy of probability of default of credit card clients. *Expert Systems
with Applications, 36*(2), 2473–2480.
https://doi.org/10.1016/j.eswa.2007.12.020

Yeh, I. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository.
https://doi.org/10.24432/C55S3H

## AI-use disclosure

The notebook includes an AI-use disclosure. AI tools were used to assist with code
debugging, explaining concepts, interpreting results, checking methodology, and
developing the final write-up. The analysis and reported results should be reviewed
against the executed notebook.
