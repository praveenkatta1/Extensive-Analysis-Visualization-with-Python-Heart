import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


sns.set_style('whitegrid')
import warnings 
warnings.filterwarnings('ignore')

heart_image = Image.open(r'C:\Users\srisa\OneDrive\Pictures\heart stroke.jpg')
st.image(heart_image, caption="Heart Stroke Image")

# Title and introductory markdown
st.title("Extensive Analysis & Visualization with Python By SRI SAI PRAVEEN KATTA")
st.markdown("""
**Heart disease** or **Cardiovascular disease (CVD)** is a class of diseases that involve the heart or blood vessels. Cardiovascular diseases are the leading cause of death globally. This is true in all areas of the world except Africa. Together CVD resulted in 17.9 million deaths (32.1%) in 2015.  Deaths, at a given age, from CVD are more common and have been increasing in much of the developing world, while rates have declined in most of the developed world since the 1970s.

So, in this kernel, I have conducted **Exploratory Data Analysis** or **EDA** of the heart disease dataset. **Exploratory Data Analysis** or **EDA** is a critical first step in analyzing a new dataset. The primary objective of EDA is to analyze the data for distribution, outliers and anomalies in the dataset. It enable us to direct specific testing of the hypothesis. It includes analysing the data to find the distribution of data, its main characteristics, identifying patterns and visualizations.  It also provides tools for hypothesis generation by visualizing and understanding the data through graphical representation.  

I hope you learn and enjoy this kernel.

**So, your upvote would be highly appreciated.**
""")

# Table of Contents
st.markdown("""
# Table of Contents

The table of contents for this project are as follows: -

1.	[Introduction to EDA](#1)
2.	[Objectives of EDA](#2)
3.	[Types of EDA](#3)
4.  [Import libraries](#4)
5.	[Import dataset](#5)
6.	[Exploratory data analysis](#6)
      - [Check shape of the dataset](#6.1)
	  - [Preview the dataset](#6.2)
	  - [Summary of dataset](#6.3)
      - [Dataset description](#6.4)
      - [Check data types of columns](#6.5)
      - [Important points about dataset](#6.6)
      - [Statistical properties of dataset](#6.7)
      - [View column names](#6.8)
7.	[Univariate analysis](#7)
      - [Analysis of `target` feature variable](#7.1)
      - [Findings of univariate analysis](#7.2)
8.	[Bivariate analysis](#8)
      - [Estimate correlation coefficients](#8.1)
      - [Analysis of `target` and `cp` variable](#8.2)
      - [Analysis of `target` and `thalach` variable](#8.3)
      - [Findings of bivariate analysis](#8.4)
9.	[Multivariate analysis](#9)
      - [Heat Map](#9.1)
      - [Pair Plot](#9.2)
10.	[Dealing with missing values](#10)
      - [Pandas isnull() and notnull() functions](#10.1)
      - [Useful commands to detect missing values](#10.2)
11.	[Check with ASSERT statement](#11)
12.	[Outlier detection](#12)
""")

# Section 1: Introduction to EDA
st.subheader("1. Introduction to EDA")
st.markdown("""
Several questions come to mind when we come across a new dataset.  The below list shed light on some of these questions:-

•	What is the distribution of the dataset?

•	Are there any missing numerical values, outliers or anomalies in the dataset?

•	What are the underlying assumptions in the dataset?

•	Whether there exists relationships between variables in the dataset?

•	How to be sure that our dataset is ready for input in a machine learning algorithm?

•	How to select the most suitable algorithm for a given dataset?

So, how do we get answer to the above questions?
""")

# Section 2: Objectives of EDA
st.subheader("2. Objectives of EDA")
st.markdown("""
The objectives of the EDA are as follows:-

i. To get an overview of the distribution of the dataset.

ii. Check for missing numerical values, outliers or other anomalies in the dataset.

iii. Discover patterns and relationships between variables in the dataset.

iv. Check the underlying assumptions in the dataset.
""")

# Section 3: Types of EDA
st.subheader("3. Types of EDA")
st.markdown("""
EDA is generally cross-classified in two ways. First, each method is either non-graphical or graphical. Second, each method is either univariate or multivariate (usually bivariate).  The non-graphical methods provide insight into the characteristics and the distribution of the variable(s) of interest. So, non-graphical methods involve calculation of summary statistics while graphical methods include summarizing the data diagrammatically.

There are four types of exploratory data analysis (EDA) based on the above cross-classification methods. Each of these types of EDA are described below:-

#### i. Univariate non-graphical EDA

The objective of the univariate non-graphical EDA is to understand the sample distribution and also to make some initial conclusions about population distributions. Outlier detection is also a part of this analysis.

#### ii. Multivariate non-graphical EDA

Multivariate non-graphical EDA techniques show the relationship between two or more variables in the form of either cross-tabulation or statistics.

#### iii. Univariate graphical EDA

In addition to finding the various sample statistics of univariate distribution (discussed above), we also look graphically at the distribution of the sample.  The non-graphical methods are quantitative and objective. They do not give full picture of the data. Hence, we need graphical methods, which are more qualitative in nature and presents an overview of the data.

#### iv. Multivariate graphical EDA

There are several useful multivariate graphical EDA techniques, which are used to look at the distribution of multivariate data. These are as follows:-

- Side-by-Side Boxplots

- Scatterplots

- Heat Maps and 3-D Surface Plots
""")

# Section 4: Import Libraries (already done at the top)
st.subheader("4. IMPORT LIBRARIES")
st.markdown("Libraries have been imported at the start of the script.")

# Section 5: Import Dataset
st.subheader("5. IMPORT DATASET")
heart = pd.read_csv(r"C:\Users\srisa\Desktop\heart.csv")
st.write(heart.head())

# Section 6: Exploratory Data Analysis
st.subheader("6. Exploratory Data Analysis")
st.markdown("""
"Check shape of the dataset

- It is a good idea to first check the shape of the dataset."
""")
st.write(heart.shape)
st.markdown("""
Now, we can see that the dataset contains 303 instances and 14 variables.
""")
st.markdown("Preview the Dataset")
st.write(heart.head(10))

st.subheader("Summary of the dataset")
st.write(heart.info())

st.subheader("Dataset description")
st.markdown("""
- The dataset contains several columns which are as follows -

  - age : age in years
  - sex : (1 = male; 0 = female)
  - cp : chest pain type
  - trestbps : resting blood pressure (in mm Hg on admission to the hospital)
  - chol : serum cholestoral in mg/dl
  - fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
  - restecg : resting electrocardiographic results
  - thalach : maximum heart rate achieved
  - exang : exercise induced angina (1 = yes; 0 = no)
  - oldpeak : ST depression induced by exercise relative to rest
  - slope : the slope of the peak exercise ST segment
  - ca : number of major vessels (0-3) colored by flourosopy
  - thal : 3 = normal; 6 = fixed defect; 7 = reversable defect
  - target : 1 or 0
""")

st.write("Check the data types of columns")
st.markdown("""
- The above `heart.info()` command gives us the number of filled values along with the data types of columns.

- If we simply want to check the data type of a particular column, we can use the following command.
""")
st.write(heart.dtypes)

st.subheader("Important points about dataset")
st.markdown("""
- `sex` is a character variable. Its data type should be object. But it is encoded as (1 = male; 0 = female). So, its data type is given as int64.

- Same is the case with several other variables - `fbs`, `exang` and `target`.

- `fbs (fasting blood sugar)` should be a character variable as it contains only 0 and 1 as values (1 = true; 0 = false). As it contains only 0 and 1 as values, so its data type is given as int64.

- `exang (exercise induced angina)` should also be a character variable as it contains only 0 and 1 as values (1 = yes; 0 = no). It also contains only 0 and 1 as values, so its data type is given as int64.

- `target` should also be a character variable. But, it also contains 0 and 1 as values. So, its data type is given as int64.
""")

st.write("Statistical properties of dataset")
st.write(heart.describe())
st.markdown("""
#### Important points to note

- The above command `heart.describe()` helps us to view the statistical properties of numerical variables. It excludes character variables.

- If we want to view the statistical properties of character variables, we should run the following command -

     `heart.describe(include=['object'])`
     
- If we want to view the statistical properties of all the variables, we should run the following command -

     `heart.describe(include='all')`  
""")
st.write("#### View Column Names")
st.write(heart.columns)

# Section 7: Univariate Analysis
st.subheader("7. Univariate analysis")
st.subheader("Analysis of `target` feature variable")
st.markdown("""
- Our feature variable of interest is `target`.

- It refers to the presence of heart disease in the patient.

- It is integer valued as it contains two integers 0 and 1 - (0 stands for absence of heart disease and 1 for presence of heart disease).

- So, in this section, I will analyze the `target` variable.
""")
st.markdown("#### Check the number of unique values in `target` variable")
st.write(heart["target"].nunique())
st.write(heart["target"].unique())
st.markdown("#### Frequency distribution of `target` variable")
st.write(heart["target"].value_counts())
st.markdown("""
- `1` stands for presence of heart disease. So, there are 165 patients suffering from heart disease.

- Similarly, `0` stands for absence of heart disease. So, there are 138 patients who do not have any heart disease.

- We can visualize this information below.
""")
st.markdown("#### Visualize frequency distribution of `target` variable")
fig, ax = plt.subplots(figsize=(12, 5))
sns.countplot(x="target", data=heart, ax=ax)
st.pyplot(fig)
st.markdown("""
#### Interpretation

- The above plot confirms the findings that -

   - There are 165 patients suffering from heart disease, and 
   
   - There are 138 patients who do not have any heart disease.
""")
st.markdown("#### Frequency distribution of `target` variable wrt `sex`")
st.write(heart.groupby('sex')['target'].value_counts())
st.markdown("""
- `sex` variable contains two integer values 1 and 0 : (1 = male; 0 = female).

- `target` variable also contains two integer values 1 and 0 : (1 = Presence of heart disease; 0 = Absence of heart disease)

- So, out of 96 females - 72 have heart disease and 24 do not have heart disease.

- Similarly, out of 207 males - 93 have heart disease and 114 do not have heart disease.

- We can visualize this information below.
""")
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x="sex", hue="target", data=heart, ax=ax)
st.pyplot(fig)
st.markdown("""
#### Interpretation

- We can see that the values of `target` variable are plotted wrt `sex` : (1 = male; 0 = female).

- `target` variable also contains two integer values 1 and 0 : (1 = Presence of heart disease; 0 = Absence of heart disease)

- The above plot confirms our findings that -

    - Out of 96 females - 72 have heart disease and 24 do not have heart disease.

    - Similarly, out of 207 males - 93 have heart disease and 114 do not have heart disease.
""")
fig = sns.catplot(x="target", col="sex", data=heart, kind="count", height=5, aspect=1)
st.pyplot(fig.figure)
st.markdown("""
- The above plot segregate the values of `target` variable and plot on two different columns labelled as (sex = 0, sex = 1).

- I think it is more convinient way of interpret the plots.
We can plot the bars horizontally as follows :
""")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(y="target", hue="sex", data=heart, ax=ax)
st.pyplot(fig)
st.markdown("We can use a different color palette as follows :")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="target", data=heart, palette="Set1", ax=ax)
st.pyplot(fig)
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="target", data=heart, palette="Set3", ax=ax)
st.pyplot(fig)
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="target", data=heart, facecolor=(0, 0, 0, 0), linewidth=5, edgecolor=sns.color_palette("dark", 3), ax=ax)
st.pyplot(fig)
st.markdown("""
- I have visualize the `target` values distribution wrt `sex`. 

- We can follow the same principles and visualize the `target` values distribution wrt `fbs (fasting blood sugar)` and `exang (exercise induced angina)`.
""")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="target", hue="fbs", data=heart, ax=ax)
st.pyplot(fig)
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="target", hue="exang", data=heart, ax=ax)
st.pyplot(fig)

st.subheader("Findings of Univariate Analysis")
st.markdown("""
Findings of univariate analysis are as follows:-

-	Our feature variable of interest is `target`.

-   It refers to the presence of heart disease in the patient.

-   It is integer valued as it contains two integers 0 and 1 - (0 stands for absence of heart disease and 1 for presence of heart disease).

- `1` stands for presence of heart disease. So, there are 165 patients suffering from heart disease.

- Similarly, `0` stands for absence of heart disease. So, there are 138 patients who do not have any heart disease.

- There are 165 patients suffering from heart disease, and 
   
- There are 138 patients who do not have any heart disease.

- Out of 96 females - 72 have heart disease and 24 do not have heart disease.

- Similarly, out of 207 males - 93 have heart disease and 114 do not have heart disease.
""")

# Section 8: Bivariate Analysis
st.subheader("8. Bivariate Analysis")
st.subheader("Estimate correlation coefficients")
st.markdown("""
Our dataset is very small. So, I will compute the standard correlation coefficient (also called Pearson's r) between every pair of attributes. I will compute it using the `heart.corr()` method as follows:-
""")
correlation = heart.corr()
st.write(correlation)
st.markdown("""
The target variable is `target`. So, we should check how each attribute correlates with the `target` variable.
""")
st.write(correlation['target'].sort_values(ascending=False))
st.markdown("""
#### Interpretation of correlation coefficient

- The correlation coefficient ranges from -1 to +1. 

- When it is close to +1, this signifies that there is a strong positive correlation. So, we can see that there is no variable which has strong positive correlation with `target` variable.

- When it is close to -1, it means that there is a strong negative correlation. So, we can see that there is no variable which has strong negative correlation with `target` variable.

- When it is close to 0, it means that there is no correlation. So, there is no correlation between `target` and `fbs`.

- We can see that the `cp` and `thalach` variables are mildly positively correlated with `target` variable. So, I will analyze the interaction between these features and `target` variable.
""")

st.subheader("Analysis of `target` and `cp` variable")
st.markdown("#### Explore `cp` variable")
st.markdown("""
- `cp` stands for chest pain type.

- First, I will check number of unique values in `cp` variable.
""")
st.write(heart['cp'].nunique())
st.markdown("So, there are 4 unique values in `cp` variable. Hence, it is a categorical variable.")
st.write(heart['cp'].value_counts())
st.markdown("""
- It can be seen that `cp` is a categorical variable and it contains 4 types of values - 0, 1, 2 and 3.
""")
st.markdown("#### Visualize the frequency distribution of `cp` variable")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="cp", data=heart, hue='cp', ax=ax)
st.pyplot(fig)
st.write(heart.groupby('cp')['target'].value_counts())
st.markdown("""
- `cp` variable contains four integer values 0, 1, 2 and 3.

- `target` variable contains two integer values 1 and 0 : (1 = Presence of heart disease; 0 = Absence of heart disease)

- So, the above analysis gives `target` variable values categorized into presence and absence of heart disease and groupby `cp` variable values.

- We can visualize this information below.
""")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="cp", hue="target", data=heart, ax=ax)
st.pyplot(fig)
st.markdown("""
- We can see that the values of `target` variable are plotted wrt `cp`.

- `target` variable contains two integer values 1 and 0 : (1 = Presence of heart disease; 0 = Absence of heart disease)

- The above plot confirms our above findings,
""")
fig = sns.catplot(x="target", col="cp", data=heart, kind="count", height=8, aspect=1)
st.pyplot(fig.figure)

st.subheader("Analysis of `target` and `thalach` variable")
st.markdown("#### Explore `thalach` variable")
st.markdown("""
- `thalach` stands for maximum heart rate achieved.

- I will check number of unique values in `thalach` variable
""")
st.write(heart['thalach'].nunique())
st.markdown("- So, number of unique values in `thalach` variable is 91. Hence, it is numerical variable.")
st.write(heart['thalach'].unique())
fig, ax = plt.subplots(figsize=(10, 6))
x = heart['thalach']
sns.distplot(x, bins=10, ax=ax)
st.pyplot(fig)
st.markdown("We can use Pandas series object to get an informative axis label :")
fig, ax = plt.subplots(figsize=(10, 6))
x = pd.Series(heart['thalach'], name="thalach variable")
sns.distplot(x, bins=10, ax=ax)
st.pyplot(fig)
st.markdown("We can plot the distribution on the vertical")
fig, ax = plt.subplots(figsize=(10, 6))
sns.distplot(heart['thalach'], bins=10, vertical=True, ax=ax)
st.pyplot(fig)
st.markdown("""
#### Seaborn Kernel Density Estimation (KDE) Plot

- The kernel density estimate (KDE) plot is a useful tool for plotting the shape of a distribution.

- The KDE plot plots the density of observations on one axis with height along the other axis.

- We can plot a KDE plot as follows :
""")
fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(heart['thalach'], ax=ax)
ax.set_title('Kernel Density Estimate of Thalach')
ax.set_xlabel('Thalach')
ax.set_ylabel('Density')
st.pyplot(fig)
st.markdown("We can shade under the density curve and use a different color")
fig, ax = plt.subplots(figsize=(10, 6))
x = pd.Series(heart['thalach'], name="thalach variable")
sns.kdeplot(x, shade=True, color='r', ax=ax)
st.pyplot(fig)
st.markdown("""
#### Histogram

- A histogram represents the distribution of data by forming bins along the range of the data and then drawing bars to show the number of observations that fall in each bin.

- We can plot a histogram as follows :
""")
fig, ax = plt.subplots(figsize=(10, 6))
sns.distplot(heart['thalach'], kde=False, rug=True, bins=10, ax=ax)
st.pyplot(fig)
fig, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="thalach", data=heart, ax=ax)
st.pyplot(fig)
st.markdown("""
- We can see that those people suffering from heart disease (target = 1) have relatively higher heart rate (thalach) as compared to people who are not suffering from heart disease (target = 0).
We can add jitter to bring out the distribution
""")
fig, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="thalach", data=heart, jitter=0.01, ax=ax)
st.pyplot(fig)
st.markdown("#### Visualize distribution of `thalach` variable wrt `target` with boxplot")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="target", y="thalach", data=heart, ax=ax)
st.pyplot(fig)
st.markdown("""
#### Interpretation

The above boxplot confirms our finding that people suffering from heart disease (target = 1) have relatively higher heart rate (thalach) as compared to people who are not suffering from heart disease (target = 0).
""")

st.subheader("Findings of Bivariate Analysis")
st.markdown("""
Findings of Bivariate Analysis are –

- There is no variable which has strong positive correlation with `target` variable.

- There is no variable which has strong negative correlation with `target` variable.

- There is no correlation between `target` and `fbs`.

- The `cp` and `thalach` variables are mildly positively correlated with `target` variable. 

- We can see that the `thalach` variable is slightly negatively skewed.

- The people suffering from heart disease (target = 1) have relatively higher heart rate (thalach) as compared to people who are not suffering from heart disease (target = 0).

- The people suffering from heart disease (target = 1) have relatively higher heart rate (thalach) as compared to people who are not suffering from heart disease (target = 0).
""")

# Section 9: Multivariate Analysis
st.subheader("9. Multivariate analysis")
st.markdown("""
- The objective of the multivariate analysis is to discover patterns and relationships in the dataset.
""")
st.subheader("Discover patterns and relationships")
st.markdown("""
- An important step in EDA is to discover patterns and relationships between variables in the dataset. 

- I will use `heat map` and `pair plot` to discover the patterns and relationships in the dataset.

- First of all, I will draw a `heat map`.
""")
fig = plt.figure(figsize=(16, 12))
plt.title('Correlation Heatmap of Heart Disease Dataset')
sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
plt.xticks(rotation=90)
plt.yticks(rotation=30)
st.pyplot(fig)
st.markdown("""
#### Interpretation

From the above correlation heat map, we can conclude that :-

- `target` and `cp` variable are mildly positively correlated (correlation coefficient = 0.43).

- `target` and `thalach` variable are also mildly positively correlated (correlation coefficient = 0.42).

- `target` and `slope` variable are weakly positively correlated (correlation coefficient = 0.35).

- `target` and `exang` variable are mildly negatively correlated (correlation coefficient = -0.44).

- `target` and `oldpeak` variable are also mildly negatively correlated (correlation coefficient = -0.43).

- `target` and `ca` variable are weakly negatively correlated (correlation coefficient = -0.39).

- `target` and `thal` variable are also weakly negatively correlated (correlation coefficient = -0.34).
""")

st.subheader("Pair Plot")
num = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
fig = sns.pairplot(heart[num], kind='scatter', diag_kind='hist')
st.pyplot(fig.figure)
st.markdown("""
- I have defined a variable `num_var`. Here `age`, `trestbps`, `chol`, `thalach` and `oldpeak` are numerical variables and `target` is the categorical variable.

- So, I will check relationships between these variables.
""")

st.subheader("Analysis of `age` and other variables")
st.markdown("#### Check the number of unique values in `age` variable")
st.write(heart['age'].nunique())
st.markdown("## View statistical summary of `age` variable##")
st.write(heart['age'].describe())
st.markdown("""
#### Interpretation

- The mean value of the `age` variable is 54.37 years.

- The minimum and maximum values of `age` are 29 and 77 years.
""")
st.markdown("#### Plot the distribution of `age` variable")
st.markdown("Now, I will plot the distribution of `age` variable to view the statistical properties.")
fig, ax = plt.subplots(figsize=(10, 6))
sns.distplot(heart['age'], bins=10, ax=ax)
st.pyplot(fig)
st.markdown("""
#### Interpretation

- The `age` variable distribution is approximately normal.
""")
st.markdown("### Analyze `age` and `target` variable")
st.markdown("#### Visualize frequency distribution of `age` variable wrt `target`")
fig, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="age", data=heart, ax=ax)
st.pyplot(fig)
st.markdown("""
#### Interpretation

- We can see that the people suffering from heart disease (target = 1) and people who are not suffering from heart disease (target = 0) have comparable ages.
""")
st.markdown("#### Visualize distribution of `age` variable wrt `target` with boxplot")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="target", y="age", data=heart, ax=ax)
st.pyplot(fig)
st.markdown("""
#### Interpretation

- The above boxplot tells two different things :

  - The mean age of the people who have heart disease is less than the mean age of the people who do not have heart disease.
  
  - The dispersion or spread of age of the people who have heart disease is greater than the dispersion or spread of age of the people who do not have heart disease.
""")
st.markdown("### Analyze `age` and `trestbps` variable")
st.markdown("I will plot a scatterplot to visualize the relationship between `age` and `trestbps` variable.")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="age", y="trestbps", data=heart, ax=ax)
st.pyplot(fig)
st.markdown("""
#### Interpretation

- The above scatter plot shows that there is no correlation between `age` and `trestbps` variable.
""")
st.markdown("### Analyze `age` and `chol` variable")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="age", y="chol", data=heart, ax=ax)
st.pyplot(fig)
fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x="age", y="chol", data=heart, ax=ax)
st.pyplot(fig)
st.markdown("""
#### Interpretation

- The above plot confirms that there is a slightly positive correlation between `age` and `chol` variables.
""")
st.markdown("### Analyze `chol` and `thalach` variable")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="chol", y="thalach", data=heart, ax=ax)
st.pyplot(fig)
fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x="chol", y="thalach", data=heart, ax=ax)
st.pyplot(fig)
st.markdown("""
#### Interpretation

- The above plot shows that there is no correlation between `chol` and `thalach` variable.
""")

# Section 10: Dealing with Missing Values
st.subheader("10. Dealing with missing values")
st.markdown("""
- In Pandas missing data is represented by two values:

  - **None**: None is a Python singleton object that is often used for missing data in Python code.
  
  - **NaN** : NaN (an acronym for Not a Number), is a special floating-point value recognized by all systems that use the standard IEEE floating-point representation.

- There are different methods in place on how to detect missing values.
""")
st.subheader("Pandas isnull() and notnull() functions")
st.markdown("""
- Pandas offers two functions to test for missing data - `isnull()` and `notnull()`. These are simple functions that return a boolean value indicating whether the passed in argument value is in fact missing data.

- Below, I will list some useful commands to deal with missing values.
""")
st.subheader("Useful commands to detect missing values")
st.markdown("""
- **heart.isnull()**

The above command checks whether each cell in a dataframe contains missing values or not. If the cell contains missing value, it returns True otherwise it returns False.

- **heart.isnull().sum()**

The above command returns total number of missing values in each column in the dataframe.

- **heart.isnull().sum().sum()** 

It returns total number of missing values in the dataframe.

- **heart.isnull().mean()**

It returns percentage of missing values in each column in the dataframe.

- **heart.isnull().any()**

It checks which column has null values and which has not. The columns which has null values returns TRUE and FALSE otherwise.

- **heart.isnull().any().any()**

It returns a boolean value indicating whether the dataframe has missing values or not. If dataframe contains missing values it returns TRUE and FALSE otherwise.

- **heart.isnull().values.any()**

It checks whether a particular column has missing values or not. If the column contains missing values, then it returns TRUE otherwise FALSE.

- **heart.isnull().values.sum()**

It returns the total number of missing values in the dataframe.
""")
st.write(heart.isnull().sum())
st.markdown("#### We can see that there are no missing values in the dataset.")

# Section 11: Check with ASSERT statement
st.subheader("11. Check with ASSERT statement")
st.markdown("""
- We must confirm that our dataset has no missing values. 

- We can write an **assert statement** to verify this. 

- We can use an assert statement to programmatically check that no missing, unexpected 0 or negative values are present. 

- This gives us confidence that our code is running properly.

- **Assert statement** will return nothing if the value being tested is true and will throw an AssertionError if the value is false.

- **Asserts**

  - assert 1 == 1 (return Nothing if the value is True)

  - assert 1 == 2 (return AssertionError if the value is False)
""")
assert pd.notnull(heart).all().all()  # assert that there are no missing values in the dataframe
assert (heart >= 0).all().all()      # assert all values are greater than or equal to 0
st.markdown("""
- The above two commands do not throw any error. Hence, it is confirmed that there are no missing or negative values in the dataset. 

- All the values are greater than or equal to zero.
""")

# Section 12: Outlier Detection
st.subheader("12. Outlier detection")
st.markdown("""
I will make boxplots to visualise outliers in the continuous numerical variables : -

`age`, `trestbps`, `chol`, `thalach` and  `oldpeak` variables.
""")
st.markdown("### `age` variable")
st.write(heart['age'].describe())
st.markdown("#### Box-plot of `age` variable")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=heart["age"], ax=ax)
st.pyplot(fig)
st.markdown("### `trestbps` variable")
st.write(heart['trestbps'].describe())
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=heart["trestbps"], ax=ax)
st.pyplot(fig)
st.markdown("### `chol` variable")
st.write(heart['chol'].describe())
st.markdown("#### Box-plot of `chol` variable")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=heart["chol"], ax=ax)
st.pyplot(fig)
st.markdown("### `thalach` variable")
st.write(heart['thalach'].describe())
st.markdown("#### Box-plot of `thalach` variable")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=heart["thalach"], ax=ax)
st.pyplot(fig)
st.markdown("### `oldpeak` variable")
st.write(heart['oldpeak'].describe())
st.markdown("#### Box-plot of `oldpeak` variable")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=heart["oldpeak"], ax=ax)
st.pyplot(fig)
st.markdown("""
#### Findings

- The `age` variable does not contain any outlier.

- `trestbps` variable contains outliers to the right side.

- `chol` variable also contains outliers to the right side.

- `thalach` variable contains a single outlier to the left side.

- `oldpeak` variable contains outliers to the right side.

- Those variables containing outliers needs further investigation.
""")

# Display the heart image at the end
st.image(heart_image, caption="Heart Stroke Image")