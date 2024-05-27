# Email analysis project
This project involves analyzing a dataset of emails to identify and categorize spam, extract key topics from spam emails, and recognize organizations mentioned in non-spam emails. The main objectives of this project are:

- Train a classifier to identify SPAM emails.
- Identify the main topics among the SPAM emails.
- Calculate the semantic distance between the identified topics to deduce heterogeneity.
- Extract and analyze organizations mentioned in NON-SPAM emails.

### Dataset
The dataset used in this project is a collection of emails from the Enron email dataset. The dataset includes the following columns:

- label: Indicates whether the email is SPAM or HAM (not spam).
- text: The content of the email.
- label_num: Numerical representation of the label (0 for HAM, 1 for SPAM).

## Project steps
### Exploratory Data Analysis
- Initial Analysis: Loading the dataset, checking for missing values, and understanding the distribution of SPAM vs. HAM emails.
- Data Visualization: Generating word clouds and distribution plots for email lengths and common words.

### Spam classification
- Preprocessing: Cleaning the text data by removing punctuation, numbers, and stopwords, and applying lemmatization.
- Model Training: Using a Random Forest classifier and SVM to identify SPAM emails. Hyperparameter tuning is performed using RandomizedSearchCV.
- Evaluation: Confusion matrix and classification report are used to evaluate the performance of the classifiers.

### Topic modeling
- Preprocessing: Cleaning SPAM email text data.
- LDA Modeling: Identifying the main topics in SPAM emails using Latent Dirichlet Allocation (LDA).
- Optimal Number of Topics: Determining the optimal number of topics using coherence scores.

### Semantic distance calculation
- Jensen-Shannon Divergence: Calculating the Jensen-Shannon distance between topic distributions to measure heterogeneity.
- Visualization: Plotting the heatmap of Jensen-Shannon distances between topics.

### Organization extraction
- Entity Extraction: Using spaCy's Named Entity Recognition (NER) to extract organizations from non-spam emails.
- Frequency Analysis: Counting the occurrences of each organization and visualizing the most frequent ones.
