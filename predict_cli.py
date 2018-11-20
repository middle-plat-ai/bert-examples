import sys
import time
import numpy as np
from ner_predict import prepare_model
from ner_predict import make_prediction

predict_file_path = './NERdata/to_predict.txt'
labels = ['NaN', 'B-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'X']


def prepare_predict_file(sentence):
    tokens = sentence.strip().split(' ')
    # end sentence with dot
    if tokens[-1] != '.':
        tokens.append('.')

    with open(predict_file_path, 'w') as f:
        # Write file header
        f.write('-DOCSTART- -X- -X- O\n')
        f.write('\n')

        for t in tokens:
            f.write('{} O O O\n'.format(t))

        f.write('\n')

    return tokens


def print_result(result, tokens):
    to_remove_keys = []
    for k, v in enumerate(result):
        if v == 10:
            to_remove_keys.append(k)

    for k in to_remove_keys:
        result = np.delete(result, k)

    for k, v in enumerate(result):
        if v != 0 and k < len(tokens):
            print('{}({}): {}({})'.format(tokens[k - 1], k, labels[v], v))


if __name__ == '__main__':

    print('Please wait while the Fine-Tuned BERT model is loaded. ')
    processor, label_list, tokenizer, estimator = prepare_model()
    print('*** Model is successfully loaded ***')

    while True:
        sentence = input('Please enter a sentence to detect: \n')
        if sentence.lower() in ['cancel', 'terminate', 'exit']:
            sys.exit()

        tokens = prepare_predict_file(sentence)
        # print('You have entered: {}'.format(sentence))
        start_time = time.time()
        result = make_prediction(processor, label_list, tokenizer, estimator)
        print_result(result, tokens)
        print('***** Prediciton takes {} seconds.'.format(str(round(time.time() - start_time, 2))))