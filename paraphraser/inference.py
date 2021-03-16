import tensorflow as tf
from embeddings import load_sentence_embeddings
from preprocess_data import preprocess_batch
from six.moves import input
from lstm_model import lstm_model
import numpy as np
from pprint import pprint as pp
import os

prefix = 'augmentation/datasets/'

domains = {
  'indomain': ['nat_questions', 'newsqa', 'squad'],
  'oodomain': ['duorc', 'race', 'relation_extraction']
}

classifications = ['train', 'val']

class Paraphraser(object):
    '''Heart of the paraphraser model.  This class loads the checkpoint
    into the Tensorflow runtime environment and is responsible for inference.
    Greedy and sampling based approaches are supported
    '''

    def __init__(self, checkpoint):
        """Constructor.  Load vocabulary index, start token, end token, unk id,
        mask_id.  Restore checkpoint.

        Args:
            checkpoint: A path to the checkpoint
        """
        self.word_to_id, self.idx_to_word, self.embedding, self.start_id, self.end_id, self.unk_id, self.mask_id = load_sentence_embeddings()
        self.checkpoint = checkpoint
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.model = lstm_model(self.sess, 'infer', 300, self.embedding, self.start_id, self.end_id, self.mask_id)
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint)

    def sample_paraphrase(self, sentence, sampling_temp=1.0, how_many=1):
        """Paraphrase by sampling a distribution

        Args:
            sentence (str): A sentence input that will be paraphrased by 
                sampling from distribution.
            sampling_temp (int) : A number between 0 an 1

        Returns:
            str: a candidate paraphrase of the `sentence`
        """

        return self.infer(1, sentence, self.idx_to_word, sampling_temp, how_many)

    def greedy_paraphrase(self, sentence):
        """Paraphrase using greedy sampler
    
        Args:
            sentence : The source sentence to be paraphrased.

        Returns:
            str : a candidate paraphrase of the `sentence`
        """

        return self.infer(0, sentence, self.idx_to_word, 0., 1)


    def infer(self, decoder, source_sent, id_to_vocab, temp, how_many):
        """ Perform inferencing.  In other words, generate a paraphrase
        for the source sentence.

        Args:
            decoder : 0 for greedy, 1 for sampling
            source_sent : source sentence to generate a paraphrase for
            id_to_vocab : dict of vocabulary index to word
            end_id : the end token
            temp : the sampling temperature to use when `decoder` is 1

        Returns:
            str : for the generated paraphrase
        """

        seq_source_words, seq_source_ids = preprocess_batch([ source_sent ] * how_many)
        #print(seq_source_words)
        #print(seq_source_ids)
        seq_source_len = [ len(seq_source) for seq_source in seq_source_ids ]
        #print(seq_source_len)

        feed_dict = {
            self.model['seq_source_ids']: seq_source_ids,
            self.model['seq_source_lengths']: seq_source_len,
            self.model['decoder_technique']: decoder,
            self.model['sampling_temperature']: temp
        }

        feeds = [
            self.model['predictions']
            #model['final_sequence_lengths']
        ]

        predictions = self.sess.run(feeds, feed_dict)[0]
        #print(predictions)
        return self.translate(predictions, decoder, id_to_vocab, seq_source_words[0])

    def translate(self, predictions, decoder, id_to_vocab, seq_source_words):
        """ Translate the vocabulary ids in `predictions` to actual words
        that compose the paraphrase.

        Args:
            predictions : arrays of vocabulary ids
            decoder : 0 for greedy, 1 for sample, 2 for beam
            id_to_vocab : dict of vocabulary index to word

        Returns:
            str : the paraphrase
        """
        translated_predictions = []
        #np_end = np.where(translated_predictions == end_id)
        for sent_pred in predictions:
            translated = []
            for pred in sent_pred:
                word = 'UUNNKK'
                if pred == self.end_id:
                    break
                if pred == self.unk_id:
                    # Search for rare word
                    for seq_source_word in seq_source_words:
                        if seq_source_word not in self.word_to_id:
                            word = seq_source_word
                else:
                    word = id_to_vocab[pred]
                translated.append(word)
            translated_predictions.append(' '.join(translated))
        return translated_predictions

def getParaphrases(paraphraser, input_path, temp, numPP):
    output_path = input_path + '-pps-all'
    if not os.path.exists(os.path.dirname(output_path)):
        try:
            os.makedirs(os.path.dirname(output_path))
        except:
            print ("Failed making directory")

    output_file = open(output_path, 'w')
    input_file = open(input_path, 'r')

    print("Paraphrasing questions in the '{}' file.".format(input_path))
    dictionary = {}
    for line in input_file.readlines():
        paraphrases = paraphraser.sample_paraphrase(line, sampling_temp=temp, how_many=numPP)
        output_file.write(line)
        for pp in paraphrases:
            pp_word_list = pp.split()
            no_break_word_list = [item for item in pp_word_list if item != '\n']
            output_file.write(" ".join(no_break_word_list) + "\n")
    input_file.close()
    output_file.close()
          

def getParaphrasesForAllDatasets(paraphraser, temp, numPP): 
    print("Starting to get paraphrases.")
    for inOrOut in domains:
        for dataset in domains[inOrOut]:
            for classification in classifications:
                input_path = prefix + inOrOut + '_' + classification + '/' + dataset + '-questions'
                getParaphrases(paraphraser, input_path, temp, numPP)
    print("Finished.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
    parser.add_argument('--useFiles', type=bool, help='Should use the default files and save the output to the file?', default=False)
    parser.add_argument('--sampleTemp', type=float, help='Sampling Temperature', default=0.75)
    parser.add_argument('--numPP', type=bool, help='Number of paraphrases to generate', default=10)
    args = parser.parse_args()
    paraphraser = Paraphraser(args.checkpoint)

    if (args.useFiles):
        return getParaphrasesForAllDatasets(paraphraser, args.sampleTemp, args.numPP)

    while 1:
        source_sentence = input("Source: ")
        #p = paraphraser.greedy_paraphrase(source_sentence)
        #print(p)
        paraphrases = paraphraser.sample_paraphrase(source_sentence, sampling_temp=0.75, how_many=10)
        for i, paraphrase in enumerate(paraphrases):
            print("Paraph #{}: {}".format(i, paraphrase))

if __name__ == '__main__':
    main()

