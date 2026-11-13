# 🏦 Personal Loan Campaign: AllLife Bank

Predicting which liability (depositor) customers are most likely to accept a personal loan, using **Decision Tree classification** with pre-pruning and post-pruning.

## 💼 Business Context

AllLife Bank is a US bank with a growing customer base. Most customers are **liability customers** (depositors), while the number of **asset customers** (borrowers) is small. The bank wants to convert more depositors into personal loan customers while retaining them as depositors.

Last year's campaign had a conversion rate of just over **9%**. The retail marketing department now wants better-targeted campaigns to improve that rate.

---

## 🎯 Objectives

As a Data Scientist at AllLife Bank, build a model that:
1. Helps marketing **identify customers with a higher probability of buying a personal loan**
2. **Predicts** whether a liability customer will accept a personal loan
3. Identifies **which customer segments to target more**

---

## 📂 Dataset

- **File:** `Loan_Modelling.csv`
- **Size:** 5,000 rows × 14 columns (13 after dropping `ID`)
- **Missing values:** none
- **Duplicates:** none

| Feature | Description |
|---|---|
| `ID` | Customer ID |
| `Age` | Customer's age in completed years |
| `Experience` | Years of professional experience |
| `Income` | Annual income (in $ thousands) |
| `ZIPCode` | Home address ZIP code |
| `Family` | Family size of the customer |
| `CCAvg` | Average monthly credit card spending (in $ thousands) |
| `Education` | 1: Undergrad, 2: Graduate, 3: Advanced/Professional |
| `Mortgage` | Value of house mortgage, if any (in $ thousands) |
| `Personal_Loan` | **Target.** Did the customer accept the personal loan in the last campaign? |
| `Securities_Account` | Has a securities account with the bank? |
| `CD_Account` | Has a certificate of deposit (CD) account with the bank? |
| `Online` | Uses internet banking? |
| `CreditCard` | Uses a credit card issued by another bank? |

---

## 🔄 Project Workflow

1. **Data Overview:** shape, data types, missing values, duplicates, statistical summary
2. **Data Preprocessing**
   - Dropped the `ID` column (no predictive value)
   - Fixed negative `Experience` values (−1, −2, −3), treated as data-entry errors
   - Mapped `Education` codes to labels (Undergraduate / Graduate / Professional)
3. **Feature Engineering**
   - Reduced `ZIPCode` from **467 unique values to 7** by keeping the first two digits
   - Converted categorical columns to the `category` dtype
4. **Exploratory Data Analysis:** univariate and bivariate analysis, correlation heatmap, pair plots
5. **Outlier Check (IQR):** outliers in `Income` (1.92%), `CCAvg` (6.48%) and `Mortgage` (5.82%) were **kept**, since they reflect real customer behaviour
6. **Data Preparation**
   - Dropped `Experience` (perfectly correlated with `Age`)
   - One-hot encoded `ZIPCode` and `Education`
   - 75/25 train-test split (`random_state=43`)
7. **Model Building:** three Decision Tree variants, evaluated mainly on **Recall**
8. **Model Comparison and Final Selection**

---

## 🔍 Key EDA Findings

- Only about **9.6%** of customers accepted the loan in the last campaign, so the target is imbalanced.
- `Income` and `CCAvg` are **right-skewed**.
- `Age` and `Experience` are **perfectly correlated**, so one was dropped.
- `Income` and `CCAvg` are moderately correlated.
- About **60%** of customers use internet banking.
- The average customer's annual income is about **$74K**, and ages range from 23 to 67.

---

## 🤖 Models and Results

### Why Recall?
Two kinds of errors are possible:
- **False Positive:** predicting a customer will take a loan when they won't. This wastes marketing resources.
- **False Negative:** predicting a customer won't take a loan when they would. This loses a potential customer.

The bigger loss is the **False Negative**, so the models were evaluated with **Recall** as the primary metric.

### Test Set Performance

| Model | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| Decision Tree (default) | 0.9792 | 0.8630 | 0.9545 | 0.9065 |
| Decision Tree (pre-pruning) | 0.9600 | **0.9110** | 0.7824 | 0.8418 |
| Decision Tree (post-pruning) | **0.9864** | 0.8904 | **0.9924** | **0.9386** |

### Training Set Performance

| Model | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| Decision Tree (default) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree (pre-pruning) | 0.7843 | 1.0000 | 0.2922 | 0.4523 |
| Decision Tree (post-pruning) | 0.9872 | 0.9102 | 0.9441 | 0.9268 |

### Observations
- The **default tree** overfits: it scores 100% on training data and is very complex and hard to interpret.
- The **pre-pruned tree** (`max_depth=2`, `max_leaf_nodes=50`, `min_samples_split=10`, `class_weight='balanced'`) gives the **highest recall (~91%)**. Its precision is lower, so it flags more false positives.
- The **post-pruned tree** (cost-complexity pruning) gives the best **accuracy, precision and F1**, with a small drop in recall.
- Pick the pre-pruned model if missing a potential customer is the main concern, or the post-pruned model if you also want to limit wasted outreach.

### Top Features (Decision Tree)
`Income` · `Education` · `Family` · `CCAvg` · `CD_Account`

---

## 💡 Business Insights and Recommendations

- **Target higher-income customers.** Income is the most important predictor. In the pruned tree, customers with income above about **$92.5K** are far more likely to accept a loan.
- **Target more educated customers.** Graduate and Professional customers are more likely to accept than Undergraduates.
- **Target larger families.** Families of **3 or more** members have a higher chance of taking a loan.
- **Cross-sell to CD account holders.** About 50% of customers with a CD account were interested in a personal loan, so they are a strong segment.
- **Deploy the model** to score the liability customer base and focus marketing spend on high-probability customers, which should raise the campaign conversion rate above 9%.

---

## 🛠 Tech Stack

- **Language:** Python
- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Machine learning:** `scikit-learn` (DecisionTreeClassifier, cost-complexity pruning, metrics)

---

## 📁 Repository Structure

```
├── Personal_Loan_Campaign.ipynb   # Main analysis and modeling notebook
├── Loan_Modelling.csv             # Dataset
└── README.md                      # Project documentation
```

---

## 👤 Author

**Your Name**
[LinkedIn](https://www.linkedin.com/in/karanmeher/) · [GitHub](https://github.com/karan-meher)
