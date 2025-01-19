from mbert import encode_data_and_get_lemma_embeddings, encode_data_and_compute_noun_embeddings, create_data_loader, clear_old_model, train_and_evaluate, eval_test_model
from utils import calculate_gender_percentages, split_dataset, plot_cm_per_seed
from get_dictionaries import create_sentence_noun_gender_dict_anim
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
import scipy.stats as stats

script_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(script_dir), 'training_results_RU_animacy/')
os.makedirs(RESULTS_DIR, exist_ok=True)

# get dictionaries
russian_dict_anim, russian_dict_inan = create_sentence_noun_gender_dict_anim('Russian')

print(len(russian_dict_anim))
print(len(russian_dict_inan))

# undersamle towards the minority class
sample_size = len(russian_dict_anim)
russian_dict_inan = random.sample(russian_dict_inan, sample_size)

print(len(russian_dict_anim))
print(len(russian_dict_inan))

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = BertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
print(device)
batch_size = 32

# Initialize the dictionary to hold data loaders for all combinations
data_loader_dict_combos = {}

combinations = ['inanim_lemma', 'inanim_context', 'anim_lemma', 'anim_context']

############### Create dataloaders for each combination ##################

for combo in combinations:
    print(f"Creating data loaders for {combo}")
    animacy, noun_form = combo.split('_')

    # Split dataset based on animacy
    if animacy == 'anim':
        train_X, train_y, train_nouns, train_lemmas, val_X, val_y, val_nouns, val_lemmas, test_X, test_y, test_nouns, test_lemmas = split_dataset(russian_dict_anim)
    else:  # inanimacy
        train_X, train_y, train_nouns, train_lemmas, val_X, val_y, val_nouns, val_lemmas, test_X, test_y, test_nouns, test_lemmas = split_dataset(russian_dict_inan)

    if noun_form == 'lemma':
        train_noun_embeddings = encode_data_and_get_lemma_embeddings(tokenizer, train_lemmas, bert_model)
        val_noun_embeddings = encode_data_and_get_lemma_embeddings(tokenizer, val_lemmas, bert_model)
        test_noun_embeddings = encode_data_and_get_lemma_embeddings(tokenizer, test_lemmas, bert_model)
    else:  # context
        train_noun_embeddings = encode_data_and_compute_noun_embeddings(tokenizer, train_X, train_nouns, bert_model, device, batch_size)
        val_noun_embeddings = encode_data_and_compute_noun_embeddings(tokenizer, val_X, val_nouns, bert_model, device, batch_size)
        test_noun_embeddings = encode_data_and_compute_noun_embeddings(tokenizer, test_X, test_nouns, bert_model, device, batch_size)

    # Create data loaders for each split (train, validation, test)
    train_data_loader = create_data_loader(train_noun_embeddings, train_y, device, batch_size)
    val_data_loader = create_data_loader(val_noun_embeddings, val_y, device, batch_size)
    test_data_loader = create_data_loader(test_noun_embeddings, test_y, device, batch_size)

    # Store the data loaders in the dictionary under the current combination
    data_loader_dict_combos[combo] = [train_data_loader, val_data_loader, test_data_loader]


########### Training and evaluation ##############

selected_layers = list(range(0, 13))
seed_values = [42, 123, 456]

combo_results = {}

for combo in combinations:
    print(f"Training for {combo}")
    animacy, noun_form = combo.split('_')

    if animacy == 'anim':
        gender_percentages = calculate_gender_percentages(russian_dict_anim)
        p_masc, p_fem, p_neut, p_common = gender_percentages

    else:
        gender_percentages = calculate_gender_percentages(russian_dict_inan)
        p_masc, p_fem, p_neut, p_common = gender_percentages

    # define random guessing baseline (like in Veeman et al. 2020)
    random_guess_baseline = (p_masc * p_masc) + (p_fem * p_fem) + (p_neut * p_neut) + (p_common * p_common)

    train_data_loader, val_data_loader, test_data_loader = data_loader_dict_combos[combo] # get data loaders

    # define result dir and plot dir
    RESULTS_DIR_COMBO = os.path.join(RESULTS_DIR, f'{combo}')
    os.makedirs(RESULTS_DIR_COMBO, exist_ok=True)
    plot_dir = os.path.join(RESULTS_DIR_COMBO, f'plots_{combo}')
    os.makedirs(plot_dir, exist_ok=True)

    results_test_dict = {
            'combo': combo,
            'gender_percentages': (p_masc, p_fem, p_neut, p_common),
            'accuracy_baseline': random_guess_baseline,
            'layers': {layer: {'accuracy_per_seed': [], 'F1_per_seed': []} for layer in selected_layers}}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.tensor(gender_percentages).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    for layer in selected_layers:
        for seed_value in seed_values:
            torch.manual_seed(seed_value)
            random.seed(seed_value)
            np.random.seed(seed_value)

            print(f"Training for layer {layer} with seed {seed_value}:")
            current_model, train_f1, val_f1 = train_and_evaluate(layer, seed_value, weights, train_data_loader, val_data_loader, plot_dir)

            # evaluate on test set
            test_loss, test_acc, test_f1, test_predictions, test_true_labels = eval_test_model(current_model, test_data_loader, loss_fn, device)

            plot_cm_per_seed(test_true_labels, test_predictions, layer, seed_value, "Russian", plot_dir)
            # Store test results per seed
            results_test_dict['layers'][layer]['F1_per_seed'].append(test_f1)
            results_test_dict['layers'][layer]['accuracy_per_seed'].append(test_acc)

            # Clear the model to free up memory
            clear_old_model(current_model)

    for layer in selected_layers:
        layer_results = results_test_dict['layers'][layer]
        layer_results['avg_F1'] = np.mean(layer_results['F1_per_seed'])
        layer_results['avg_accuracy'] = np.mean(layer_results['accuracy_per_seed'])

    results = {
        f"gender_percentages": results_test_dict['gender_percentages'],
        "random_guessing_baseline": results_test_dict['accuracy_baseline'],
        "test_acc": [results_test_dict['layers'][layer]['avg_accuracy'] for layer in selected_layers],
        "test_f1": [results_test_dict['layers'][layer]['avg_F1'] for layer in selected_layers]
    }

    result_filename = f"results_mBERT_layer_RU_{combo}.json"
    result_filepath = os.path.join(RESULTS_DIR_COMBO, result_filename)

    combo_results[combo] = results

result_filename = f"results_mBERT_layer_RU_animacy.json"
result_filepath = os.path.join(os.path.dirname(RESULTS_DIR), result_filename)

with open(result_filepath, "w") as f:
    json.dump(combo_results, f, indent=4)

print(f"Results saved to {result_filepath}")

######### Plotting ############

categories = ['Animate Lemma', 'Inanimate Lemma', 'Animate Contextualized', 'Inanimate Contextualized']

dataframes = []

# Extract F1 scores per layer and per combination
for combo, results in combo_results.items():
    for layer in selected_layers:
        # Get average F1 score for each layer and combo
        avg_f1 = results['test_f1'][selected_layers.index(layer)]

        category_mapping = {
            'anim_lemma': 'Animate Lemma',
            'inanim_lemma': 'Inanimate Lemma',
            'anim_context': 'Animate Contextualized',
            'inanim_context': 'Inanimate Contextualized'
        }
        category = category_mapping.get(combo, 'Unknown')
        dataframes.append({
            "Layer": layer,
            "F1 Score": avg_f1,
            "Category": category
        })

# Convert the list of dictionaries to a DataFrame
df_f1_scores = pd.DataFrame(dataframes)

# Custom colors for each category
color_palette = {
    'Animate Lemma': '#F5A7C0',  # Light pink
    'Inanimate Lemma': '#ADD8E6',  # Light blue
    'Animate Contextualized': '#D15B8A',  # Dark pink
    'Inanimate Contextualized': '#4682B4'  # Blue
}

plt.figure(figsize=(12, 8))
sns.lineplot(data=df_f1_scores, x="Layer", y="F1 Score", hue="Category", marker="o", palette=color_palette)
plt.title(f"F1 Scores per Layer for Different Combinations (Animate vs Inanimate, Lemma vs Contextualized)")
plt.xlabel("Layer")
plt.ylabel("F1 Score")
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

plot_path = os.path.join(RESULTS_DIR, f"test_f1s_mBERT_layer_RU_animacy_all_combos.png")
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")


###### Wilcoxon Signed-Rank Test ############

animate_lemma_f1 = []
inanimate_lemma_f1 = []
animate_context_f1 = []
inanimate_context_f1 = []

# Group F1 scores by layer and category
for combo, results in combo_results.items():
    for layer in selected_layers:
        avg_f1 = results['test_f1'][selected_layers.index(layer)]

        if combo == 'anim_lemma':
            animate_lemma_f1.append(avg_f1)
        elif combo == 'inanim_lemma':
            inanimate_lemma_f1.append(avg_f1)
        elif combo == 'anim_context':
            animate_context_f1.append(avg_f1)
        elif combo == 'inanim_context':
            inanimate_context_f1.append(avg_f1)

# Ensure that each list has the same number of elements (one per layer)
assert len(animate_lemma_f1) == len(inanimate_lemma_f1) == len(animate_context_f1) == len(inanimate_context_f1)

# 1. Compare Animate Lemma vs Inanimate Lemma using Wilcoxon
stat_lemma, p_value_lemma = stats.wilcoxon(animate_lemma_f1, inanimate_lemma_f1)

# 2. Compare Animate Contextualized vs Inanimate Contextualized using Wilcoxon
stat_context, p_value_context = stats.wilcoxon(animate_context_f1, inanimate_context_f1)

print("Statistical Test Results:")

# Animate Lemma vs Inanimate Lemma
print("\nComparison: Animate Lemma vs Inanimate Lemma")
print(f"Statistic: {stat_lemma:.4f}, P-value: {p_value_lemma:.4f}")
if p_value_lemma < 0.05:
    print("The difference between Animate Lemma and Inanimate Lemma is statistically significant.")
else:
    print("The difference between Animate Lemma and Inanimate Lemma is not statistically significant.")

# Animate Contextualized vs Inanimate Contextualized
print("\nComparison: Animate Contextualized vs Inanimate Contextualized")
print(f"Statistic: {stat_context:.4f}, P-value: {p_value_context:.4f}")
if p_value_context < 0.05:
    print("The difference between Animate Contextualized and Inanimate Contextualized is statistically significant.")
else:
    print(
        "The difference between Animate Contextualized and Inanimate Contextualized is not statistically significant.")



