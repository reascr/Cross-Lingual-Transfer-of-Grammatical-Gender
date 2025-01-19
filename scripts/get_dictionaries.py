import tarfile
from conllu import parse_incr
import os
from tqdm import tqdm
import pickle
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path_in = os.path.join(os.path.dirname(script_dir), 'data', 'raw_UD_data')
data_path_out = os.path.join(os.path.dirname(script_dir),'data/dictionaries')
#treebank_tar = os.path.join(data_path_in, 'ud-treebanks-v2.14.tgz')

# extract treebank dataset
#with tarfile.open(treebank_tar, 'r:gz') as tar:
#    tar.extractall(path=data_path_in)


def get_language_folders(language):
    '''Return the folder names for the given language from the dataset.'''
    language = language.capitalize()
    extracted_path = os.path.join(data_path_in, 'ud-treebanks-v2.14')
    folders = [folder for folder in os.listdir(extracted_path) if folder.startswith(f'UD_{language}-')]
    return folders


def get_conllu_filepaths(language_folders):
    '''Get .conllu files from all language folders for the language'''
    filepaths = []
    for folder in language_folders:
        folder_path = os.path.join(data_path_in, 'ud-treebanks-v2.14', folder)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.conllu'):
                filepaths.append(os.path.join(folder_path, file_name))
    return filepaths


def create_sentence_noun_gender_dict(language, max_sample_size = 60000):
    """Create a dictionary of sentences, nouns, and their gender."""
    sentence_noun_gender_dict = []
    language_folders = get_language_folders(language)
    filepaths = get_conllu_filepaths(language_folders)

    for filepath in tqdm(filepaths, desc=f"Processing files for {language}", unit="file"):
        with open(filepath, 'r', encoding='utf-8') as f:
            sentences = parse_incr(f)  # parse the conllu file incrementally
            for tokenlist in sentences:
                sentence = " ".join(token['form'] for token in tokenlist)  # Original sentence
                for idx, token in enumerate(tokenlist):
                    if token['upostag'] == 'NOUN':
                        noun = token['form']
                        lemma = token['lemma']
                        feats = token.get('feats')
                        if feats is not None:
                            gender = feats.get('Gender', None)  # Extract gender
                            if gender in ['Masc', 'Fem', 'Neut', 'Com']:
                                sentence_noun_gender_dict.append({
                                    'sent': sentence,  # Full sentence
                                    'noun': noun,  # Direct token form
                                    'lemma': lemma, # Lemma or Base form
                                    'gender': gender  # Gender of the noun
                                })

    # remove sentences with multiple occurrences of the target noun
    print(f"Extracted {len(sentence_noun_gender_dict)} for {language}. Filtering sentences now with several occurrences of target noun.\n")

    processed_results = []
    for entry in sentence_noun_gender_dict:
        noun = entry['noun']
        sentence = entry['sent']
        # Count occurrences of the noun in the original sentence
        if sentence.count(noun) == 1:  # Only proceed if the noun appears exactly once
            processed_results.append({'sent': sentence, 'noun': noun, 'lemma': entry['lemma'], 'gender': entry['gender']})

    # reduce size of dictionary for computational reasons
    if len(processed_results) > max_sample_size:
        print(f"Number of sentences in dict > {max_sample_size}. Reducing to {max_sample_size}.")
        processed_results = random.sample(processed_results, max_sample_size)

    sentence_noun_gender_dict = processed_results
    return sentence_noun_gender_dict


def create_sentence_noun_gender_dict_anim(language, max_sample_size=60000):
    """Create two dictionary of sentences, nouns, and their gender for animate and inanimate nouns. If the language has the distinction Human, Inhuman, Inanimate, the former to get combined to Animate."""
    sentence_noun_gender_dict = []
    language_folders = get_language_folders(language)
    print(language_folders, len(language_folders))
    filepaths = get_conllu_filepaths(language_folders)

    for filepath in tqdm(filepaths, desc=f"Processing files for {language}", unit="file"):
        with open(filepath, 'r', encoding='utf-8') as f:
            sentences = parse_incr(f)  # parse the conllu file incrementally
            for tokenlist in sentences:
                sentence = " ".join(token['form'] for token in tokenlist)  # Original sentence
                for idx, token in enumerate(tokenlist):
                    if token['upostag'] == 'NOUN':
                        noun = token['form']
                        lemma = token['lemma']
                        feats = token.get('feats')

                        if feats is not None:
                            gender = feats.get('Gender', None)  # Extract gender
                            animacy = feats.get('Animacy', None)  # Extract animacy
                            if gender in ['Masc', 'Fem', 'Neut', 'Com'] and animacy in ['Inan', 'Anim', 'Hum', 'Nhum']:
                                sentence_noun_gender_dict.append({
                                    'sent': sentence,  # Full sentence
                                    'noun': noun,  # Direct token form
                                    'lemma': lemma,  # Lemma of noun
                                    'gender': gender,  # Gender of the noun
                                    'animacy': animacy # Animacy of noun
                                })
    # Filter out sentences with multiple occurrences of the target noun
    print(f"Extracted {len(sentence_noun_gender_dict)} entries for {language}. Filtering out sentences with multiple occurrences of target nouns.\n")

    filtered_results = []
    for entry in sentence_noun_gender_dict:
        noun = entry['noun']
        sentence = entry['sent']
        # Only include if the noun appears exactly once in the sentence
        if sentence.count(noun) == 1:
            filtered_results.append(entry)

    # split the list into animate and inanimate nouns TO DO: here I should iterate over filtered_results....
    animacy_dict = [entry for entry in sentence_noun_gender_dict if
                         entry['animacy'] in ['Anim', 'Hum', 'Inhum']]  # merge Human and Inhum to anim
    inanim_dict = [entry for entry in sentence_noun_gender_dict if entry['animacy'] == 'Inan']

    if len(animacy_dict) > max_sample_size:
        print(f"Animate dictionary size exceeds {max_sample_size}. Reducing to {max_sample_size}.")
        animacy_dict = random.sample(animacy_dict, max_sample_size)

    if len(inanim_dict) > max_sample_size:
        print(f"Inanimate dictionary size exceeds {max_sample_size}. Reducing to {max_sample_size}.")
        inanim_dict = random.sample(inanim_dict, max_sample_size)

    return animacy_dict, inanim_dict



def save_dict_to_pickle(dictionary, language):
    """Save the dictionary to a .pkl file."""
    file_path = os.path.join(os.path.dirname(script_dir),'data', 'dictionaries', f'{language}_noun_gender_dict.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(dictionary, f)
    print(f"{language} dictionary saved to {file_path}")

if __name__ == "__main__":
    languages = ['German', 'Greek', 'Russian', 'Italian', 'Danish', 'Arabic', 'Beja']

    for language in languages:
        noun_gender_dict = create_sentence_noun_gender_dict(language)
        save_dict_to_pickle(noun_gender_dict, language)
        print(f"Dictionary size for {language}: {len(noun_gender_dict)}")
        print(noun_gender_dict[0])
