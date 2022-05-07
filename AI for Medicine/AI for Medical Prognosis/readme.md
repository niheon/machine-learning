# AI for Medical Prognosis

<img src="https://www.itp.net/cloud/2021/08/19/AI-3-2.jpg" alt="AI for Medical Prognosis">

The ability to carry out daily activities; the risk for complications and associated health conditions; and the likelihood of survival are all part of Prognosis. The normal course of the detected disease, the individual's physical and mental health, the available therapies, and other considerations are used to make a prognosis. The projected duration, function, and description of the disease's history, such as steady decline, intermittent crisis, or sudden, unpredictable crisis, are all included in a full Prognosis.

## Table of Contents

- [Build and Evaluate a Linear Risk model](https://github.com/rajeshai/machine-learning/blob/main/AI%20for%20Medicine/AI%20for%20Medical%20Prognosis/Part%201%20-%20Build%20and%20Evaluate%20a%20Linear%20Risk%20model.ipynb)
- [Risk Models Using Tree-based Models](https://github.com/rajeshai/machine-learning/blob/main/AI%20for%20Medicine/AI%20for%20Medical%20Prognosis/Part%202%20-%20Risk%20Models%20Using%20Tree-based%20Models.ipynb)
- [Survival Estimates that Vary with Time](https://github.com/rajeshai/machine-learning/blob/main/AI%20for%20Medicine/AI%20for%20Medical%20Prognosis/Part%203%20-%20Survival%20Estimates%20that%20Vary%20with%20Time.ipynb)
- [Cox Proportional Hazards and Random Survival](https://github.com/rajeshai/machine-learning/blob/main/AI%20for%20Medicine/AI%20for%20Medical%20Prognosis/Part%204%20-%20Cox%20Proportional%20Hazards%20and%20Random%20Survival%20Forests.ipynb)

## Build and Evaluate a Linear Risk model

The name given to general practice in applied statistics, biostatistics, econometrics, and other related disciplines of generating an easily calculated number (the score) that reflects the level of risk in the presence of several risk factors is risk score (or risk scoring) (e.g. risk of mortality or disease in the presence of symptoms or genetic profile, risk financial loss considering credit and financial history, etc.).

In this project a risk score model for retinopathy in diabetes patients using logistic regression is built. Retinopathy is an eye condition that causes changes to the blood vessels in the part of the eye called the retina. This often leads to vision changes or blindness. Diabetic patients are known to be at high risk for retinopathy.

Logistic regression is an appropriate analysis to use for predicting the probability of a binary outcome. In our case, this would be the probability of having or not having diabetic retinopathy.

## Risk Models Using Tree-based Models

In this project, risk models are built using tree based models like Decision Trees and Random Forests. The task is to predict the 10-year risk of death of individuals from the NHANES I epidemiology dataset (for a detailed description of this dataset you can check the [CDC Website](https://wwwn.cdc.gov/nchs/nhanes/nhefs/default.aspx/). 

The following steps are followed in this project
- Dealing with Missing Data
  -* Complete Case Analysis
  - Imputation
- Decision Trees
  - Evaluation
  - Regularization
- Random Forests
  - Hyperparameter Tuning
