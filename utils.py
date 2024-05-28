import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter
import string
from nltk.util import ngrams
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
from gensim import corpora
import gensim
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import numpy as np
from scipy.spatial.distance import jensenshannon
from gensim.matutils import kullback_leibler, sparse2full
import spacy
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def load_dataset(file_path):
    """
    Load the dataset from the given file path.

    Parameters:
    file_path (str): The URL or local path to the dataset CSV file.

    Returns:
    pd.DataFrame: The loaded dataset.
    """
    df = pd.read_csv(file_path)
    return df

def display_basic_info(df):
    """
    Display basic information about the dataset, including info, head, missing values,
    target variable distribution, duplicate rows, and check if all emails start with "Subject".

    Parameters:
    df (pd.DataFrame): The dataset.
    """
    # Display basic information about the dataset
    print("Dataset Information:")
    df.info()
    print(df.head())

    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Display the distribution of the target variable (SPAM/Not SPAM)
    print("\nDistribution of SPAM vs Not SPAM:")
    print(df['label'].value_counts())

    # Check for duplicate rows
    print("\nNumber of duplicate rows:")
    print(df.duplicated().sum())

    # Check if all mail start with "Subject"
    all_start_with_subject = df['text'].str.startswith('Subject').all()
    print("\nAll emails start with 'Subject':", all_start_with_subject)
def preprocess_dataframe(df):
    """
    Preprocess the dataframe by dropping unnecessary columns and cleaning text data.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: The preprocessed dataframe.
    """
    # Drop the 'Unnamed: 0' column
    df = df.drop(columns=['Unnamed: 0'])

    # Remove the word "Subject:" from the beginning of each email
    df['text'] = df['text'].str.replace(r'^Subject: ', '', regex=True)
    
    return df

def plot_label_distribution(df):
    """
    Plot the distribution of SPAM vs HAM emails.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='label', palette='viridis')
    plt.title('Distribution of SPAM vs HAM Emails')
    plt.xlabel('Email Type')
    plt.ylabel('Count')
    plt.show()

def generate_wordclouds(df):
    """
    Generate and plot word clouds for SPAM and HAM emails.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    """
    spam_text = ' '.join(df[df['label'] == 'spam']['text'])
    ham_text = ' '.join(df[df['label'] == 'ham']['text'])

    spam_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(spam_text)
    ham_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(ham_text)

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(spam_wordcloud, interpolation='bilinear')
    plt.title('Word Cloud for SPAM Emails')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(ham_wordcloud, interpolation='bilinear')
    plt.title('Word Cloud for HAM Emails')
    plt.axis('off')

    plt.show()

def plot_email_length_distribution(df):
    """
    Plot the distribution of email lengths for SPAM and HAM emails.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    """
    df['length'] = df['text'].apply(len)

    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='length', hue='label', multiple='stack', bins=50, palette='viridis')
    plt.title('Distribution of Email Lengths')
    plt.xlabel('Email Length')
    plt.ylabel('Count')
    plt.show()

def preprocess_text_for_common_words(text):
    """
    Preprocess text by converting to lowercase, removing punctuation, and stopwords.

    Parameters:
    text (str): The input text.

    Returns:
    list: A list of preprocessed words.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return words

def plot_common_words(df):
    """
    Plot the most common words in SPAM and HAM emails.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    """
    df['processed_text'] = df['text'].apply(preprocess_text_for_common_words)

    spam_words = [word for words in df[df['label'] == 'spam']['processed_text'] for word in words]
    ham_words = [word for words in df[df['label'] == 'ham']['processed_text'] for word in words]

    spam_common_words = Counter(spam_words).most_common(20)
    ham_common_words = Counter(ham_words).most_common(20)

    spam_common_df = pd.DataFrame(spam_common_words, columns=['word', 'count'])
    ham_common_df = pd.DataFrame(ham_common_words, columns=['word', 'count'])

    plt.figure(figsize=(12, 6))
    sns.barplot(data=spam_common_df, x='count', y='word', palette='viridis')
    plt.title('Most Common Words in SPAM Emails')
    plt.xlabel('Count')
    plt.ylabel('Word')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=ham_common_df, x='count', y='word', palette='viridis')
    plt.title('Most Common Words in HAM Emails')
    plt.xlabel('Count')
    plt.ylabel('Word')
    plt.show()
def get_top_bigrams(text, n=None):
    """
    Get the top bigrams from a list of words.

    Parameters:
    text (list): The input text as a list of words.
    n (int): The number of top bigrams to return. Default is None.

    Returns:
    list: A list of bigrams.
    """
    bigrams = ngrams(text, 2)
    return [' '.join(bigram) for bigram in bigrams]

def extract_bigrams(df):
    """
    Extract and count the most common bigrams in SPAM and HAM emails.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame, pd.DataFrame: DataFrames containing the most common bigrams for SPAM and HAM emails.
    """
    spam_bigrams = [bigram for text in df[df['label'] == 'spam']['processed_text'] for bigram in get_top_bigrams(text)]
    ham_bigrams = [bigram for text in df[df['label'] == 'ham']['processed_text'] for bigram in get_top_bigrams(text)]

    spam_common_bigrams = Counter(spam_bigrams).most_common(20)
    ham_common_bigrams = Counter(ham_bigrams).most_common(20)

    spam_bigrams_df = pd.DataFrame(spam_common_bigrams, columns=['bigram', 'count'])
    ham_bigrams_df = pd.DataFrame(ham_common_bigrams, columns=['bigram', 'count'])

    return spam_bigrams_df, ham_bigrams_df

def plot_bigrams(bigrams_df, title):
    """
    Plot the most common bigrams.

    Parameters:
    bigrams_df (pd.DataFrame): DataFrame containing bigrams and their counts.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(data=bigrams_df, x='count', y='bigram', palette='viridis')
    plt.title(title)
    plt.xlabel('Count')
    plt.ylabel('Bigram')
    plt.show()
def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing punctuation and numbers,
    tokenizing, removing stopwords, and lemmatizing.

    Parameters:
    text (str): The input text.

    Returns:
    str: The preprocessed text.
    """
    # Remove the word "Subject:" from the beginning of each email
    text = re.sub(r'^Subject: ', '', text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r'[^\w\s]|[\d]', '', text)
    tokens = text.split()
    stopwords_set = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stopwords_set and len(token) > 2]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

def apply_text_preprocessing(df):
    """
    Apply text preprocessing to the 'text' column of the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: The dataframe with an additional 'cleaned_text' column containing preprocessed text.
    """
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    return df

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier using RandomizedSearchCV for hyperparameter tuning.

    Parameters:
    X_train (pd.Series): The training data.
    y_train (pd.Series): The training labels.

    Returns:
    RandomizedSearchCV: The trained RandomizedSearchCV object.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    param_distributions = {
        'rf__n_estimators': [50, 100, 200, 500],
        'rf__max_features': ['sqrt', 'log2'],
        'rf__max_depth': [None, 10, 20, 30, 40, 50],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }

    random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=50, cv=5, verbose=3, n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    return random_search

def train_svm(X_train, y_train):
    """
    Train an SVM classifier using RandomizedSearchCV for hyperparameter tuning.

    Parameters:
    X_train (pd.Series): The training data.
    y_train (pd.Series): The training labels.

    Returns:
    RandomizedSearchCV: The trained RandomizedSearchCV object.
    """
    pipeline_svm = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('scaler', StandardScaler(with_mean=False)),
        ('svm', SVC(random_state=42))
    ])

    param_distributions_svm = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': [1, 0.1, 0.01, 0.001],
        'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    random_search_svm = RandomizedSearchCV(pipeline_svm, param_distributions_svm, n_iter=50, cv=5, verbose=3, n_jobs=-1, random_state=42)
    random_search_svm.fit(X_train, y_train)
    return random_search_svm

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print the classification report and confusion matrix.

    Parameters:
    model (RandomizedSearchCV): The trained RandomizedSearchCV object.
    X_test (pd.Series): The testing data.
    y_test (pd.Series): The testing labels.

    Returns:
    dict: The classification report as a dictionary.
    np.array: The confusion matrix.
    """
    y_pred = model.predict(X_test)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Best Hyperparameters found:")
    print(model.best_params_)

    print("Classification Report:")
    print(pd.DataFrame(classification_rep).transpose())

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    return classification_rep, conf_matrix
def create_combined_classification_report(classification_report_dict):
    """
    Create a combined classification report DataFrame for multiple models.

    Parameters:
    classification_report_dict (dict): Dictionary containing classification reports for different models.

    Returns:
    pd.DataFrame, pd.DataFrame: Combined classification report DataFrame and accuracy DataFrame.
    """
    # Create DataFrames from the classification report dictionary
    rf_report_df = pd.DataFrame(classification_report_dict['Random Forest']).transpose()
    svm_report_df = pd.DataFrame(classification_report_dict['SVM']).transpose()

    # Remove "macro avg" and "weighted avg" rows
    rf_report_df = rf_report_df.drop(['macro avg', 'weighted avg', 'accuracy'])
    svm_report_df = svm_report_df.drop(['macro avg', 'weighted avg', 'accuracy'])

    # Add a column to identify the model
    rf_report_df['Model'] = 'Random Forest'
    svm_report_df['Model'] = 'SVM'

    # Combine the DataFrames
    combined_report_df = pd.concat([rf_report_df, svm_report_df])

    # Reorganize the columns for better layout
    combined_report_df.reset_index(inplace=True)
    combined_report_df = combined_report_df.rename(columns={'index': 'Metric'})
    combined_report_df = combined_report_df.pivot(index='Metric', columns='Model', values=['precision', 'recall', 'f1-score'])
    combined_report_df.columns = [f'{metric}_{model}' for metric, model in combined_report_df.columns]
    combined_report_df.reset_index(inplace=True)

    # Create a separate table for accuracy
    accuracy_df = pd.DataFrame({
        'Model': ['Random Forest', 'SVM'],
        'Accuracy': [classification_report_dict['Random Forest']['accuracy'], classification_report_dict['SVM']['accuracy']]
    })

    return combined_report_df, accuracy_df
def preprocess_spam_text(text):
    """
    Preprocess spam email text by converting to lowercase, removing punctuation and numbers,
    tokenizing, removing stopwords, and lemmatizing.

    Parameters:
    text (str): The input text.

    Returns:
    list: The preprocessed tokens.
    """
    # Remove the word "Subject:" from the beginning of each email
    text = re.sub(r'^Subject: ', '', text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    stopwords_set = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords_set and len(token) > 2]
    return tokens

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute coherence values for different numbers of topics.

    Parameters:
    dictionary (gensim.corpora.Dictionary): The dictionary created from the texts.
    corpus (list of list of (int, int)): The corpus created from the texts.
    texts (list of list of str): The preprocessed texts.
    limit (int): The maximum number of topics.
    start (int): The minimum number of topics.
    step (int): The step size for the number of topics.

    Returns:
    list: List of LDA models.
    list: Coherence values for each number of topics.
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def find_optimal_number_of_topics(coherence_values, start=2, step=3):
    """
    Find the optimal number of topics based on coherence values.

    Parameters:
    coherence_values (list): List of coherence values.
    start (int): The starting number of topics.
    step (int): The step size for the number of topics.

    Returns:
    int: The optimal number of topics.
    """
    x = range(start, start + step * len(coherence_values), step)
    optimal_num_topics = x[coherence_values.index(max(coherence_values))]
    return optimal_num_topics

def plot_coherence_values(coherence_values, start=2, step=3, limit=12):
    """
    Plot coherence values for different numbers of topics.

    Parameters:
    coherence_values (list): List of coherence values.
    start (int): The starting number of topics.
    step (int): The step size for the number of topics.
    limit (int): The maximum number of topics.
    """
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Score for Number of Topics")
    plt.show()

def get_topic_distributions(lda_model, num_topics, dictionary):
    """
    Extract topic distributions from the LDA model.

    Parameters:
    lda_model (gensim.models.ldamodel.LdaModel): The trained LDA model.
    num_topics (int): The number of topics.
    dictionary (gensim.corpora.Dictionary): The dictionary created from the texts.

    Returns:
    list: List of topic distributions.
    """
    topic_distributions = []
    for i in range(num_topics):
        topic_words = lda_model.get_topic_terms(i, topn=len(dictionary))
        topic_distributions.append(sparse2full(topic_words, len(dictionary)))
    return topic_distributions

def calculate_js_distances(topic_distributions, num_topics):
    """
    Calculate Jensen-Shannon distances between topic distributions.

    Parameters:
    topic_distributions (list): List of topic distributions.
    num_topics (int): The number of topics.

    Returns:
    np.array: Matrix of Jensen-Shannon distances.
    """
    js_distances = np.zeros((num_topics, num_topics))
    for i in range(num_topics):
        for j in range(num_topics):
            if i != j:
                js_distances[i][j] = jensenshannon(topic_distributions[i], topic_distributions[j])
            else:
                js_distances[i][j] = 0.0
    return js_distances

def plot_js_distances(js_distances, num_topics):
    """
    Plot a heatmap of Jensen-Shannon distances between topics.

    Parameters:
    js_distances (np.array): Matrix of Jensen-Shannon distances.
    num_topics (int): The number of topics.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(js_distances, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=range(1, num_topics + 1), yticklabels=range(1, num_topics + 1))
    plt.title('Heatmap of Jensen-Shannon Distances between Topics')
    plt.xlabel('Topic')
    plt.ylabel('Topic')
    plt.show()

def load_spacy_model(model_name="en_core_web_sm"):
    try:
        # Try to load the spaCy model
        nlp = spacy.load(model_name)
    except OSError:
        # If the model is not found, download it
        print(f"Downloading spaCy model: {model_name}")
        subprocess.run(["python", "-m", "spacy", "download", model_name])
        nlp = spacy.load(model_name)
    return nlp

nlp = load_spacy_model()

def extract_organizations(text):
    """
    Extract organizations from a text using spaCy NER.

    Parameters:
    text (str): The input text.

    Returns:
    list: List of extracted organizations.
    """
    doc = nlp(text)
    organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return organizations

def get_organizations_from_non_spam(df):
    """
    Extract organizations from non-spam emails and count their occurrences.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: DataFrame containing organizations and their frequencies.
    """
    # Filter non-spam emails
    non_spam_df = df[df['label'] == 'ham'].copy()
    # Apply the extraction function to non-spam emails
    non_spam_df['organizations'] = non_spam_df['text'].apply(extract_organizations)
    # Remove emails without organizations
    non_spam_df = non_spam_df[non_spam_df['organizations'].map(lambda x: len(x) > 0)]
    # Count organization occurrences
    all_organizations = [org for sublist in non_spam_df['organizations'] for org in sublist]
    organization_counts = Counter(all_organizations)
    # Convert to DataFrame for easier visualization
    org_df = pd.DataFrame(organization_counts.items(), columns=['Organization', 'Frequency'])
    org_df = org_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    return org_df

def plot_top_organizations(org_df, top_n=10):
    """
    Plot the top N most frequent organizations.

    Parameters:
    org_df (pd.DataFrame): DataFrame containing organizations and their frequencies.
    top_n (int): Number of top organizations to plot.
    """
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frequency', y='Organization', data=org_df.head(top_n))
    plt.title(f'Top {top_n} Most Frequent Organizations in Non-Spam Emails')
    plt.xlabel('Frequency')
    plt.ylabel('Organization')
    plt.show()

