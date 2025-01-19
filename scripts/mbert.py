import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


class GenderClassifier(nn.Module):
    def __init__(self, layer, num_labels=4, hidden_size=768):
        super(GenderClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)  # add classification layer on top
        self.layer = layer  # layer we want to pass to classification head

    def forward(self, noun_embeddings):
        # classify based on the noun's embedding
        selected_layer_embedding = noun_embeddings[:, self.layer, :]
        logits = self.classifier(selected_layer_embedding)
        return logits


def encode_data_and_get_lemma_embeddings(tokenizer, lemmas, bert_model, batch_size=32):
    """
    Tokenizes lemmas in batches and computes their embeddings using BERT (without sentence context).

    Parameters:
    - tokenizer: mBERT tokenizer
    - lemmas: list of lemmas to process
    - batch_size: number of lemmas to process at once

    Returns:
    - torch.Tensor: tensor of lemma embeddings [num_lemmas, num_layers, hidden_size]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lemma_embeddings = []

    for i in range(0, len(lemmas), batch_size):
        batch_lemmas = lemmas[i:i + batch_size]
        tokenized_inputs = tokenizer(batch_lemmas, return_tensors='pt', padding=True, truncation=False)
        tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}

        with torch.no_grad():
            outputs = bert_model(input_ids=tokenized_inputs['input_ids'],
                                 attention_mask=tokenized_inputs['attention_mask'],
                                 output_hidden_states=True)
            all_layer_embeddings = outputs.hidden_states  # [num_layers, batch_size, seq_len, hidden_size]
            batch_lemma_embeddings = torch.stack([layer.mean(dim=1) for layer in all_layer_embeddings])  # average across all sub tokens
            lemma_embeddings.append(batch_lemma_embeddings)

    # Concatenate along the batch dimension
    lemma_embeddings = torch.cat(lemma_embeddings, dim=1)  # [num_layers, total_lemmas, hidden_size]

    # Transpose to [total_lemmas, num_layers, hidden_size]
    lemma_embeddings = lemma_embeddings.permute(1, 0, 2)  # Swap batch_size and num_layers dimensions

    # print(lemma_embeddings.shape) # shape [num_lemmas, num_layers, hidden_size]
    return lemma_embeddings


def find_subtoken_pos_t(noun_ids, sent_ids):
    """Returns the start and end index of a noun's subtokens in the sentence's token IDs."""
    noun_length = noun_ids.size(0)

    # Loop through the sentence to find the noun
    for i in range(sent_ids.size(0) - noun_length + 1):
        if torch.equal(sent_ids[i:i + noun_length], noun_ids):
            return i, i + noun_length  # Return the start and end index (end is exclusive)
    return None


def encode_data_and_compute_noun_embeddings(tokenizer, texts, nouns, bert_model, device, batch_size=32):
    """
    Tokenizes texts in batches and computes BERT noun embeddings for each text.

    Parameters:
    - tokenizer: mBERT tokenizer
    - texts: List of sentences to process
    - nouns: List of target nouns (one for each sentence)
    - batch_size: Number of sentences to process at once

    Returns:
    - torch.Tensor: Tensor of noun embeddings [batch_size, num_layers, hidden_size]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noun_embeddings = []

    # Use tqdm to show a progress bar for batch processing
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches", unit="batch"):
        batch_texts = texts[i:i + batch_size]
        batch_nouns = nouns[i:i + batch_size]
        tokenized_inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=False)
        tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}

        with torch.no_grad():
            outputs = bert_model(input_ids=tokenized_inputs['input_ids'],
                                 attention_mask=tokenized_inputs['attention_mask'],
                                 output_hidden_states=True)
            all_layer_embeddings = outputs.hidden_states

        for j, noun in enumerate(batch_nouns):
            noun_token_ids = tokenizer.encode(noun, add_special_tokens=False)
            valid_token_ids = torch.tensor(noun_token_ids).to(device)

            subtoken_pos = find_subtoken_pos_t(valid_token_ids, tokenized_inputs['input_ids'][j])
            if subtoken_pos is None:
                continue

            subtoken_pos_start, subtoken_pos_end = subtoken_pos
            stacked_layer_embeddings = torch.stack([layer[j, subtoken_pos_start:subtoken_pos_end, :].mean(dim=0)
                                                    for layer in all_layer_embeddings])
            noun_embeddings.append(stacked_layer_embeddings)

    return torch.stack(noun_embeddings)


def create_data_loader(noun_embeddings, labels, device, batch_size=32):
    """Creates a DataLoader for noun embeddings and labels."""
    labels_tensor = torch.tensor(labels).to(device)
    dataset = TensorDataset(noun_embeddings, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size)


def clear_old_model(model):  # delete model from GPU
    model.to('cpu')
    del model
    torch.cuda.empty_cache()
    # gc.collect()

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            noun_embeddings, labels = [x.to(device) for x in batch]
            outputs = model(noun_embeddings)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # Collect predictions and true labels
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(true_labels, predictions, average='macro')
    return avg_loss, f1


def train_and_evaluate(layer_number, random_seed, weights, source_train_data_loader, source_val_data_loader, plot_dir, patience=5):
    model = GenderClassifier(num_labels=4, layer=layer_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # auslagern
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(
        weight=weights)  # weighted loss function because of the class imbalance towards feminine
    # loss_fn = nn.CrossEntropyLoss()  # loss function for multinomial logistic regression, not weighted
    optimizer = AdamW(model.parameters(), lr=5e-5)  # optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR every 10 epochs
    num_epochs = 30
    # best_val_accuracy = 0
    best_val_f1 = 0
    patience_counter = 0

    # Check if a previous model state exists and delete it
    model_filename = f"best_model_state_layer_{layer_number}.bin"
    if os.path.exists(model_filename):
        os.remove(model_filename)

    # Initialize lists to store loss and accuracy for plotting
    train_losses = []
    # train_accuracies = []
    train_f1s = []
    val_losses = []
    # val_accuracies = []
    val_f1s = []

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase with progress bar
        model.train()  # Set the model to training mode
        # total_loss, total_correct = 0, 0  # Reset loss and correct count for each epoch
        total_loss = 0
        true_labels = []
        preds = []

        for batch in tqdm(source_train_data_loader, desc="Training", unit="batch"):
            noun_embeddings, labels = [x.to(device) for x in batch]

            optimizer.zero_grad()
            outputs = model(noun_embeddings)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            batch_preds = torch.argmax(outputs, dim=1)
            preds.extend(batch_preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            # Calculate accuracy
            # preds = torch.argmax(outputs, dim=1)
            # total_correct += torch.sum(preds == labels).item()

        # Calculate average loss and accuracy for the epoch
        # avg_loss = total_loss / len(source_train_data_loader)
        # accuracy = total_correct / len(source_train_data_loader.dataset)
        # train_losses.append(avg_loss)  # Store training loss
        # train_accuracies.append(accuracy)  # Store training accuracy
        # print(f"Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}")

        # Calculate average loss and F1 score for the epoch
        avg_loss = total_loss / len(source_train_data_loader)
        train_losses.append(avg_loss)  # Store training loss
        train_f1 = f1_score(true_labels, preds, average='macro')
        train_f1s.append(train_f1)  # Store training F1 score
        print(f"Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}")

        # Validation phase
        # val_loss, val_accuracy = eval_model(model, source_val_data_loader, loss_fn, device)
        # val_losses.append(val_loss)  # Store validation loss
        # val_accuracies.append(val_accuracy)  # Store validation accuracy
        # print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Validation phase
        val_loss, val_f1 = eval_model(model, source_val_data_loader, loss_fn, device)  # Using eval_model for F1 score
        val_losses.append(val_loss)  # Store validation loss
        val_f1s.append(val_f1)  # Store validation F1 score
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        # Save model if validation accuracy improves
        # if val_accuracy > best_val_accuracy:
        #   best_val_accuracy = val_accuracy
        # torch.save(model.state_dict(), model_filename)

        # Early stopping based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_filename)  # Save the best model
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # If patience is exceeded, stop training early
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        # Update the learning rate scheduler
        scheduler.step()

        # Adjust the number of epochs for plotting based on early stopping
    actual_epochs = len(train_losses)  # The actual number of epochs trained

    # Plotting
    pastel_pink = '#FFB3BA'
    pastel_purple = '#D5C6E0'

    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, actual_epochs + 1), train_losses, label='Train Loss', marker='o', color=pastel_pink)
    plt.plot(range(1, actual_epochs + 1), val_losses, label='Validation Loss', marker='o', color=pastel_purple)
    plt.title(f'Loss Over Epochs for Layer {layer_number} and seed {random_seed}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plot training and validation F1 scores
    plt.subplot(1, 2, 2)
    plt.plot(range(1, actual_epochs + 1), train_f1s, label='Train F1', marker='o', color=pastel_pink)
    plt.plot(range(1, actual_epochs + 1), val_f1s, label='Validation F1', marker='o', color=pastel_purple)
    plt.title(f'F1 Over Epochs for Layer {layer_number} and seed {random_seed}')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    # plt.show()
    plot_filename = os.path.join(plot_dir, f'Train_Val_Loss_F1_{layer_number}_{random_seed}.png')
    plt.savefig(plot_filename)
    print(f"Combined loss and F1 plot saved to {plot_filename}")

    return model, train_f1s, val_f1s




def eval_test_model(model, data_loader, loss_fn, device):
    model.eval()  # Set the model to evaluation mode
    total_loss, total_correct = 0, 0
    predictions = []
    true_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for batch in data_loader:
            noun_embeddings, labels = [x.to(device) for x in batch]

            # Forward pass
            outputs = model(noun_embeddings)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()  # Accumulate loss

            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            # Calculate accuracy
            total_correct += torch.sum(preds == labels).item()


    # Calculate average loss and accuracy for the test set
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / len(data_loader.dataset)
    f1 = f1_score(true_labels, predictions, average='macro')
    return avg_loss, accuracy, f1, predictions, true_labels


