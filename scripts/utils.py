from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import random
import os
import pickle as pkl

def load_language_dict(language, data_path):
    file_path = os.path.join(data_path, f"{language}_noun_gender_dict.pkl")
    with open(file_path, 'rb') as f:
        language_dict = pkl.load(f)
    return language_dict

def calculate_gender_percentages(data):
    """
    Given the data, calculate the percentage distribution of each gender.

    Parameters:
    - data: A list of dictionaries containing gender information

    Returns:
    - masc, fem, neut, com percentages
    """
    gender_counts = Counter(item['gender'] for item in data)
    total = sum(gender_counts.values())
    percentages = {gender: count / total for gender, count in gender_counts.items()}
    masc_per = percentages.get('Masc', 0)
    fem_per = percentages.get('Fem', 0)
    neut_per = percentages.get('Neut', 0)
    common_per = percentages.get('Com', 0)
    return masc_per, fem_per, neut_per, common_per


def plot_gender_distribution(data, title, ax, color_mapping):
    # Count occurrences of each gender
    gender_counts = Counter(item['gender'] for item in data)

    # Calculate percentages
    total = sum(gender_counts.values())
    percentages = {gender: count / total * 100 for gender, count in gender_counts.items()}
    labels = percentages.keys()
    sizes = percentages.values()

    colors = [color_mapping.get(gender, '#FFFFFF') for gender in labels]

    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')
    ax.set_title(title)

    return percentages


def split_dataset(data, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Splits data into training, validation, and test sets.

    Parameters:
    - data: List of data
    - gender_mapping: A dictionary that maps gender labels
    - train_size: Proportion of data used for training
    - val_size: Proportion of data used for validation
    - test_size: Proportion of data used for testing

    Returns:
    - tuple: train, validation, and test sets with labels
    """

    gender_mapping = {
        'Masc': 0,
        'Fem': 1,
        'Neut': 2,
        'Com': 3
    }

    random.shuffle(data)
    train_data, rem_data = train_test_split(data, test_size=(val_size + test_size))
    val_data, test_data = train_test_split(rem_data, test_size=test_size / (val_size + test_size))

    def extract_data(data):
        sentences = [item['sent'] for item in data]
        labels = [gender_mapping[item['gender']] for item in data]
        nouns = [item['noun'] for item in data]
        lemma = [item['lemma'] for item in data]
        return sentences, labels, nouns, lemma

    train_sent, train_labels, train_nouns, train_lemmas = extract_data(train_data)
    val_sent, val_labels, val_nouns, val_lemmas = extract_data(val_data)
    test_sent, test_labels, test_nouns, test_lemmas = extract_data(test_data)

    return (train_sent, train_labels, train_nouns, train_lemmas,
            val_sent, val_labels, val_nouns, val_lemmas,
            test_sent, test_labels, test_nouns, test_lemmas)


def plot_cm(test_true_labels, test_predictions, layer, plot_dir):
    cm = confusion_matrix(test_true_labels, test_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu',
                xticklabels=['Masculine', 'Feminine', 'Neuter', 'Common'],
                yticklabels=['Masculine', 'Feminine', 'Neuter', 'Common'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix on Test Set for layer {layer}')

    # Save the plot
    os.makedirs(plot_dir, exist_ok=True)  # Ensure the directory exists
    plot_path = os.path.join(plot_dir, f"layer_{layer}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Confusion matrix plot saved to {plot_path}")
    #plt.show()


def plot_cm_per_seed(test_true_labels, test_predictions, layer, random_seed, target_language, plot_dir):
    cm = confusion_matrix(test_true_labels, test_predictions)

    # Plotting confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu',
                xticklabels=['Masculine', 'Feminine', 'Neuter', 'Common'],
                yticklabels=['Masculine', 'Feminine', 'Neuter', 'Common'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix on Test Set for layer {layer}')

    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"layer_{layer}_{random_seed}_{target_language}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Confusion matrix plot saved to {plot_path}")

