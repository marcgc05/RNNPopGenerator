from collections import Counter

#Run to generate the Vocab from tokens
def GenerateVocabTxt():
    token_counter = Counter()

    for i in range(1,910):
        filePath = f"POP909/{i:03}/tokens.txt"
        with open(filePath) as f:
            for line in f:
                token = line.strip()
                if token:
                    token_counter[token] += 1

    vocab = [token for token, _ in token_counter.most_common()]
    output_file = "POP909/vocab.txt"
    with open(output_file, "w") as f:
        for token in vocab:
            f.write(token + "\n")

    print(f"Vocabulary saved to {output_file} with {len(vocab)} tokens.")

def load_vocab():
    vocab_file="POP909/vocab.txt"
    with open(vocab_file) as f:
        vocab = [line.strip() for line in f if line.strip()]
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for i, token in enumerate(vocab)}
    return vocab, token_to_id, id_to_token

def token_sequence_to_ids(tokens, token_to_id):
    return [token_to_id[token] for token in tokens if token in token_to_id]

def ids_to_token_sequence(ids, id_to_token):
    return [id_to_token[i] for i in ids if i in id_to_token]

'''
Example usage of the vocab tools:

vocab, token_to_id, id_to_token = load_vocab()

print(ids_to_token_sequence([1,2,3,4],id_to_token))

-^ this will return : ['NOTE_ON_DEGREE_1_OCT_4_DUR_0.25', 'NOTE_ON_DEGREE_1_OCT_5_DUR_0.25', 'NOTE_ON_DEGREE_5_OCT_5_DUR_0.25', 'NOTE_ON_DEGREE_3_OCT_5_DUR_0.25']

print(token_sequence_to_ids(['NOTE_ON_DEGREE_1_OCT_4_DUR_0.25'],token_to_id))

-^ this will return : [1]
'''

