from mbert import encode_data_and_compute_noun_embeddings, create_data_loader, clear_old_model, train_and_evaluate, eval_test_model
from utils import load_language_dict, calculate_gender_percentages, split_dataset, plot_cm_per_seed

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path_dicts = os.path.join(os.path.dirname(script_dir),'data/dictionaries')
RESULTS_DIR = os.path.join(os.path.dirname(script_dir), 'training_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

files = os.listdir(data_path_dicts)
language_files = [f for f in files if '_noun_gender_dict.pkl' in f]
languages = [] # languages we have a stored dictionary for

for file in language_files:
    # Assuming the file name format is like "German_noun_gender_dict.pkl"
    language = file.split("_")[0]
    languages.append(language)


######### GETTING DICTIONARIES ##########

all_language_dicts = {} # Dictionary to store the dictionaries
language_dataset_loaders = {} # Dictionary to store the data loaders for train, val, test set for every language (for nouns in full form)
#language_dataset_loaders_lemmas = {}  # Dictionary to store the data loaders for train, val, test set for every language (for lemmas)
gender_percentages = {}  # Store gender percentages for each language (masc, fem, neut, com) for computation of baselines

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = BertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
print(device)
batch_size = 32

# Load, split, and store datasets for each language
for language in languages:
    language_dict = load_language_dict(language, data_path_dicts)
    print(f"Processing data for {language} with number of nouns = {len(language_dict)}")
    all_language_dicts[language] = language_dict
    gender_percentages[language] = calculate_gender_percentages(language_dict)

    # Split dataset for the current language
    train_X, train_y, train_nouns, train_lemmas, val_X, val_y, val_nouns, val_lemmas, test_X, test_y, test_nouns, test_lemmas = split_dataset(language_dict)

    # Create noun embeddings and data loaders for lemmas
    train_noun_embeddings = encode_data_and_compute_noun_embeddings(tokenizer, train_X, train_nouns, bert_model, device, batch_size)
    train_data_loader = create_data_loader(train_noun_embeddings, train_y, device, batch_size)

    val_noun_embeddings = encode_data_and_compute_noun_embeddings(tokenizer, val_X, val_nouns, bert_model, device, batch_size)
    val_data_loader = create_data_loader(val_noun_embeddings, val_y, device, batch_size)

    test_noun_embeddings = encode_data_and_compute_noun_embeddings(tokenizer, test_X, test_nouns, bert_model, device, batch_size)
    test_data_loader = create_data_loader(test_noun_embeddings, test_y, device, batch_size)

    language_dataset_loaders[language] = [train_data_loader, val_data_loader, test_data_loader]

    print("\n")



##################### Train and Evaluate Cross Lingual Grammatical Gender Assignment for all Source Languages ########################

selected_layers = list(range(0, 13))
seed_values = [42, 123, 456]

# Train on Source Language
for source_language in languages:
    #source_dict = all_language_dicts[source_language]
    print(f"Training for source language {source_language}")

    source_train_data_loader, source_val_data_loader, _ = language_dataset_loaders[source_language] # get dataloader for train and val set

    # Directory to save plots
    plot_dir = os.path.join(RESULTS_DIR, f'plots_{source_language}_mBERT')
    os.makedirs(plot_dir, exist_ok=True)

    source_gender_percentages = gender_percentages[source_language]
    p_masc_s, p_fem_s, p_neut_s, p_common_s = source_gender_percentages

    results_test_dict = {}  # dictionary to store, test dataloaders, gender percentages, accuracy baseline, test accuracies (per layer), and F1 (per layer) for every language
    for target_language in languages:
        target_gender_percentages = gender_percentages[target_language]
        p_masc_t, p_fem_t, p_neut_t, p_common_t = target_gender_percentages

        # define random guessing baseline (like in Veeman et al. 2020)
        random_guess_baseline = (p_masc_s * p_masc_t) + (p_fem_s * p_fem_t) + (p_neut_s * p_neut_t) + (p_common_s * p_common_t)

        _, _, test_data_loader = language_dataset_loaders[target_language]

        # Initialize target language results
        results_test_dict[target_language] = {
            'gender_percentages': target_gender_percentages,
            'accuracy_baseline': random_guess_baseline,
            'layers': {layer: {'accuracy_per_seed': [], 'F1_per_seed': []} for layer in selected_layers},
            'test_data_loader': test_data_loader
        }


    train_f1s = []
    val_f1s = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.tensor(source_gender_percentages).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    for layer in selected_layers:
        #train_f1_seed = []
        #val_f1_seed = []

        for seed_value in seed_values:
            torch.manual_seed(seed_value)
            random.seed(seed_value)
            np.random.seed(seed_value)

            print(f"Training for layer {layer} with seed {seed_value}:")
            current_model, train_f1, val_f1 = train_and_evaluate(layer, seed_value, weights, source_train_data_loader, source_val_data_loader, plot_dir)
            #train_f1_seed.append(train_f1)
            #val_f1_seed.append(val_f1)

            # evaluate on all test sets
            for target_language in languages:
                test_loader = results_test_dict[target_language]['test_data_loader']
                test_loss, test_acc, test_f1, test_predictions, test_true_labels = eval_test_model(
                    current_model, test_loader, loss_fn, device
                )

                plot_cm_per_seed(test_true_labels, test_predictions, layer, seed_value, target_language, plot_dir) # save cm

                # Store test results per seed
                results_test_dict[target_language]['layers'][layer]['F1_per_seed'].append(test_f1)
                results_test_dict[target_language]['layers'][layer]['accuracy_per_seed'].append(test_acc)

            clear_old_model(current_model)

    for target_language in languages:
        for layer in selected_layers:
            layer_results = results_test_dict[target_language]['layers'][layer]
            layer_results['avg_F1'] = np.mean(layer_results['F1_per_seed'])
            layer_results['avg_accuracy'] = np.mean(layer_results['accuracy_per_seed'])

    results = {
        "source_language": source_language,
        "target_results": {}
    }

    for target_language in languages:

        target_results = {
            f"gender_percentages": results_test_dict[target_language]['gender_percentages'],
            "random_guessing_baseline": results_test_dict[target_language]['accuracy_baseline'],
            "test_acc": [results_test_dict[target_language]['layers'][layer]['avg_accuracy'] for layer in selected_layers],
            "test_f1": [results_test_dict[target_language]['layers'][layer]['avg_F1'] for layer in selected_layers]
        }

        results["target_results"][target_language] = target_results

    result_filename = f"{source_language}_results_mBERT_layer.json"
    result_filepath = os.path.join(RESULTS_DIR, result_filename)

    with open(result_filepath, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_filepath}")

    ######### plot test accuracies for each target language #########

    dataframes = [] # Create a list to store DataFrames for each target language

    # For each target language, prepare a DataFrame
    for target_language in languages:
        # Retrieve the test accuracies for this target language per layer
        test_f1s_target = [results_test_dict[target_language]['layers'][layer]['avg_F1'] for layer in selected_layers]

        # Create the DataFrame for this target language
        df = pd.DataFrame({
            "Layer": selected_layers,
            f"Test F1": test_f1s_target,
            "Source-Target Pair": f"{source_language} --> {target_language}"  # Language pair in the legend
        })

        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    # print(combined_df)

    # Find the maximum points for each source-target pair (based on test F1)
    max_points = combined_df.loc[combined_df.groupby("Source-Target Pair")["Test F1"].idxmax()]

    pastel_colors = sns.color_palette("muted", n_colors=len(languages))
    color_palette = {lang: pastel_colors[i] for i, lang in enumerate(languages)}
    sns.set_palette([color_palette[lang.split()[-1]] for lang in languages])

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=combined_df, x="Layer", y="Test F1", hue="Source-Target Pair",  marker="o")

    # Overlay the maximum points with larger circles
    plt.scatter(
        max_points["Layer"],
        max_points["Test F1"],
        color="magenta",
        s=150,
        label="Max F1"
    )


    plt.title(f"Test F1 Scores per Layer for source language {source_language}")
    plt.xlabel("Layer")
    plt.ylabel("Test F1 Score")
    plt.legend(title="Language Pair", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plot_path = os.path.join(plot_dir, f"test_f1s_{source_language}_mBERT_layer.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    #plt.show()