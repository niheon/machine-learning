# AI for Medical Treatment

<img src="https://www.mobihealthnews.com/sites/default/files/Doctor%20using%20hologram%2C%20virtual%20screen_Mobi%20-%20Getty_MR.Cole_Photographer_compressed.jpg" alt="AI for Medical Treatment">

The management and care of a patient for the goal of combating sickness, damage, or condition is referred to as Treatment. Restrictions on activity limits are not considered treatment unless the primary goal is to improve the worker's condition through conservative care. Medical Treatment is necessary to cure the disease and take care of patients health. AI helps to monitor the health of the patient at this stage and gives the accurate condition of the patient.

## Table of Contents
- [Estimating Treatment Effect Using Machine Learning](https://github.com/rajeshai/machine-learning/blob/main/AI%20for%20Medicine/AI%20for%20Medical%20Treatment/Part%201%20-%20Estimating%20Treatment%20Effect%20Using%20Machine%20Learning.ipynb)
- [Natural Language Entity Extraction](https://github.com/rajeshai/machine-learning/blob/main/AI%20for%20Medicine/AI%20for%20Medical%20Treatment/Part%202%20-%20Natural%20Language%20Entity%20Extraction.ipynb)
- [Machine Learning Interpretation](https://github.com/rajeshai/machine-learning/blob/main/AI%20for%20Medicine/AI%20for%20Medical%20Treatment/Part%203%20-%20Machine%20Learning%20Interpretation.ipynb)

## Estimating Treatment Effect Using Machine Learning

In this project, the main objective is to examine data from an RCT, measuring the effect of a particular drug combination on colon cancer. Specifically, the effect of Levamisole and Fluorouracil on patients who have had surgery to remove their colon cancer is studied. After surgery, the curability of the patient depends on the remaining residual cancer. In this study, it was found that this particular drug combination had a clear beneficial effect, when compared with Chemotherapy.

The following concepts are extensively discussed in this project:
- How to analyze data from a randomized control trial using both:
  - traditional statistical methods
  - and the more recent machine learning techniques
- Interpreting Multivariate Models
  - Quantifying treatment effect
  - Calculating baseline risk
  - Calculating predicted risk reduction
- Evaluating Treatment Effect Models
  - Comparing predicted and empirical risk reductions
  - Computing C-statistic-for-benefit
- Interpreting ML models for Treatment Effect Estimation
  - Implement T-learner

## Natural Language Entity Extraction

In this project, we extract disease labels for patients from unstructured clinical reports. Instead of "learning" from the dataset, we will primarily build different "rules" that help us extract knowledge from natural language.Because there is less risk of overfitting when using a "rules-based" approach, we will just use one dataset which will also be the test set.

The test set consists of 1,000 X-ray reports that have been manually labeled by a board certified radiologist for the presence or lack of presence of different pathologies. We also have access to the extracted "Impression" section of each report which is the overall summary of the radiologists for each X-ray.

The following concepts are discussed:
- Extracting disease labels from clinical reports
  - Text matching
  - Evaluating a labeler
  - Negation detection
  - Dependency parsing
- Question Answering with BERT
  - Preprocessing text for input
  - Extracting answers from model output

## Machine Learning Interpretation

In this project, we focus on the interpretation of machine learning and deep learning models. The following concepts are discussed:
- Interpreting Deep Learning Models
  - Understanding output using GradCAMs
- Feature Importance in Machine Learning
  - Permutation Method
  - SHAP Values

GradCAM is a technique to visualize the impact of each region of an image on a specific output for a Convolutional Neural Network model. Through GradCAM, we can generate a heatmap by computing gradients of the specific class scores we are interested in visualizing. Perhaps the most complicated part of computing GradCAM is accessing intermediate activations in our deep learning model and computing gradients with respect to the class output.

When developing predictive models and risk measures, it's often helpful to know which features are making the most difference. This is easy to determine in simpler models such as linear models and decision trees. However as we move to more complex models to achieve high performance, we usually sacrifice some interpretability. In this project we'll try to regain some of that interpretability using Shapley values, a technique which has gained popularity in recent years, but which is based on classic results in cooperative game theory.
