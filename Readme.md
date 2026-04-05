# 📊 PreplyProjects

A curated collection of data science and machine learning projects developed in one-on-one collaboration with students on [Preply](https://preply.com). Each folder represents a real tutoring engagement — from hands-on statistical modelling and time series forecasting to classification, NLP, and deep learning.

> **All projects are written in Python using Jupyter Notebooks.**  
> Topics span the full data science pipeline: data wrangling, exploratory analysis, statistical modelling, machine learning, and deployment-ready outputs.

---

## Table of Contents

- [About](#about)
- [Repository Structure](#repository-structure)
- [Project Summaries](#project-summaries)
  - [Topic-Based Modules](#topic-based-modules)
  - [Student Collaboration Projects](#student-collaboration-projects)
- [Technologies & Libraries](#technologies--libraries)
- [How to Use This Repository](#how-to-use-this-repository)
- [About the Instructor](#about-the-instructor)

---

## About

This repository contains Jupyter Notebook projects built during tutoring sessions on Preply. The work covers a broad range of data science competencies, tailored to each student's background and goals. Projects are either:

- **Topic-driven** — structured learning modules on a specific technique (e.g., regression, time series, classification)
- **Student-driven** — end-to-end projects built around a student's own dataset or professional problem

All code is self-contained and designed to be readable and educational.

---

## Repository Structure

```
PreplyProjects/
│
├── TimeSeriesAnalysis/          # Forecasting with ARIMA, Prophet, LSTMs
├── RegressionAnalysis/          # Linear, logistic, polynomial regression
├── Classification/              # ML classification models & evaluation
├── StatisticalModelling/        # Hypothesis testing, distributions, inference
├── CaseStudiesStatistics/       # Applied statistics on real-world datasets
├── Health/                      # Healthcare data analysis & predictive modelling
├── PandasReference/             # Pandas data manipulation reference notebooks
├── TensorflowSlides/            # Deep learning with TensorFlow/Keras
├── TweepyTracker/               # Twitter/X data collection & NLP analysis
├── Ubiqum/                      # IoT / smart home energy consumption analysis
│
├── Andy/                        # Student project: Andy
├── Anglina/                     # Student project: Anglina
├── Ann/                         # Student project: Ann
├── James/                       # Student project: James
├── Jan/                         # Student project: Jan
├── Laura/                       # Student project: Laura
├── Matteo/                      # Student project: Matteo
├── Riccardo/                    # Student project: Riccardo
├── Ronald/                      # Student project: Ronald
│
└── Readme.md
```

---

## Project Summaries

### Topic-Based Modules

#### 📈 TimeSeriesAnalysis
Forecasting techniques applied to sequential data. Covers classical statistical methods and modern deep learning approaches.

Key topics:
- Stationarity testing (ADF, KPSS)
- ARIMA / SARIMA modelling with `statsmodels`
- Facebook Prophet for trend and seasonality decomposition
- LSTM-based forecasting with TensorFlow/Keras
- Model evaluation: MAE, RMSE, MAPE

---

#### 📉 RegressionAnalysis
Supervised learning for continuous target prediction.

Key topics:
- Simple and multiple linear regression
- Polynomial and ridge/lasso regularisation
- Logistic regression for binary outcomes
- Residual diagnostics and assumption checking
- Feature selection and VIF analysis

---

#### 🏷️ Classification
Machine learning models for categorical prediction tasks.

Key topics:
- Decision Trees, Random Forest, Gradient Boosting
- Support Vector Machines (SVM)
- k-Nearest Neighbours (kNN)
- Model evaluation: confusion matrix, ROC-AUC, F1-score
- Cross-validation and hyperparameter tuning with `GridSearchCV`

---

#### 📐 StatisticalModelling
Probability theory and inferential statistics with Python.

Key topics:
- Probability distributions (Normal, Binomial, Poisson)
- Hypothesis testing: t-tests, chi-squared, ANOVA
- Confidence intervals and p-value interpretation
- Bayesian vs frequentist approaches
- Correlation and covariance analysis

---

#### 🔬 CaseStudiesStatistics
Real-world datasets tackled with applied statistical methods. Each case study follows the full pipeline: problem framing → data cleaning → analysis → conclusions.

---

#### 🏥 Health
Healthcare and biomedical data analysis. Involves patient outcome prediction, clinical data EDA, and epidemiological modelling.

Key topics:
- Medical dataset preprocessing and handling missing values
- Predictive modelling for health outcomes
- Survival analysis concepts
- Visualising clinical distributions

---

#### 🐼 PandasReference
A practical reference guide for data manipulation with Pandas — ideal as a companion notebook during other projects.

Key topics:
- DataFrame creation, indexing, and slicing
- `groupby`, `merge`, `pivot_table`, `melt`
- Handling missing data (`fillna`, `dropna`, `interpolate`)
- Time-aware indexing with `DatetimeIndex`
- Vectorised string operations and `apply`/`map`

---

#### 🧠 TensorflowSlides
Deep learning fundamentals with TensorFlow 2 and Keras, structured as teaching slides for guided sessions.

Key topics:
- Neural network architecture (Dense, Conv, LSTM layers)
- Activation functions, dropout, batch normalisation
- Training loops, callbacks, and early stopping
- Image classification with CNNs
- Sequence modelling for text and time series

---

#### 🐦 TweepyTracker
Social media data collection and analysis using the Twitter/X API via `tweepy`.

Key topics:
- Authenticating and streaming tweets with Tweepy
- Keyword and hashtag tracking
- NLP preprocessing: tokenisation, stopword removal, stemming
- Sentiment analysis (VADER / TextBlob)
- Frequency analysis and word clouds

---

#### 🏠 Ubiqum
Analysis of IoT smart home energy consumption data — a classic dataset used in data science education.

Key topics:
- Multi-channel time series EDA
- Sub-metering and energy disaggregation
- Seasonality and trend decomposition
- Predictive modelling for energy demand

---

### Student Collaboration Projects

Each folder corresponds to a real student collaboration on Preply. Projects are tailored to the student's dataset, domain, and learning objectives.

| Folder | Likely Domain / Focus |
|--------|-----------------------|
| `Andy` | Data analysis or ML project |
| `Anglina` | Statistical analysis or data visualisation |
| `Ann` | Applied statistics or regression |
| `James` | Machine learning classification or regression |
| `Jan` | Time series or forecasting |
| `Laura` | Data wrangling, EDA, or modelling |
| `Matteo` | ML or statistical modelling |
| `Riccardo` | ML or statistical modelling |
| `Ronald` | End-to-end data science project |

---

## Technologies & Libraries

| Category | Libraries |
|----------|-----------|
| **Data manipulation** | `pandas`, `numpy` |
| **Visualisation** | `matplotlib`, `seaborn`, `plotly` |
| **Machine learning** | `scikit-learn` |
| **Deep learning** | `tensorflow`, `keras` |
| **Time series** | `statsmodels`, `prophet`, `pmdarima` |
| **NLP / Social media** | `tweepy`, `nltk`, `textblob`, `vaderSentiment` |
| **Statistical tests** | `scipy.stats`, `statsmodels` |
| **Environment** | Jupyter Notebook / JupyterLab |

---

## How to Use This Repository

### 1. Clone the repository

```bash
git clone https://github.com/alketcecaj12/PreplyProjects.git
cd PreplyProjects
```

### 2. Set up a Python environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Or with conda:

```bash
conda create -n preply python=3.10
conda activate preply
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow tweepy prophet jupyter
```

### 3. Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

### 4. Navigate to any folder

Open the `.ipynb` file inside any project folder and run cells sequentially. Most notebooks are self-contained — data is either loaded from a URL, a local file, or generated inline.

---

## About the Instructor

This repository was created and maintained by **Alket Cecaj**, a Data Scientist and Quantitative Analyst with a PhD in Industrial Innovation Engineering and 12+ years of experience in applied machine learning, statistical modelling, and data science education.

Alket has taught data science and programming at university level and has been tutoring students on Preply across a broad range of topics — from introductory statistics to advanced deep learning and time series forecasting.

- 🔗 GitHub: [alketcecaj12](https://github.com/alketcecaj12)
- 📚 Background: PhD (Industrial Innovation Engineering), post-doctoral research, publications in computational social science and ML

---

*If you are a current or prospective Preply student and would like to work through any of these materials, feel free to reach out via the Preply platform.*
