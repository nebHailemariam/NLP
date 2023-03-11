import sys
import re

from collections import defaultdict


class HMMTrain():
    def __init__(self, TAG_FILE, TOKEN_FILE, OUTPUT_FILE):
        self.TAG_FILE = TAG_FILE
        self.TOKEN_FILE = TOKEN_FILE 
        self.OUTPUT_FILE = OUTPUT_FILE
        #Vocabulary
        self.vocab = {}
        self.OOV_WORD = "OOV"
        self.INIT_STATE = "init"
        self.FINAL_STATE = "final"
        #Transition and emission probabilities
        self.emissions = {} # dictionary to store count of transition from POS tag state to tokens in the corpus
        self.transitions = {} # dictionary to store count of transition from previous POS tag state to current state
        self.transitions_total = defaultdict(lambda: 0) # total number of transitions from one state to all others
        self.emissions_total = defaultdict(lambda: 0) # total number of transitions from one state to all tokens



    # train the model
    def train(self):
        # Read from tag file and token file. 
        with open(self.TAG_FILE) as tag_file, open(self.TOKEN_FILE) as token_file:
            for tag_string, token_string in zip(tag_file, token_file):
                tags = re.split("\s+", tag_string.rstrip())
                tokens = re.split("\s+", token_string.rstrip())
                pairs = zip(tags, tokens)

                # Starts off with initial state
                prevtag = self.INIT_STATE

                for (tag, token) in pairs:

                    # this block is a little trick to help with out-of-vocabulary (OOV)
                    # words.  the first time we see *any* word token, we pretend it
                    # is an OOV.  this lets our model decide the rate at which new
                    # words of each POS-type should be expected (e.g., high for nouns,
                    # low for determiners).

                    if token not in self.vocab:
                        self.vocab[token] = 1
                        token = self.OOV_WORD

                    #=======================TO-DO=======================
                     
                    if(tag not in self.emissions):
                        # initialize to store count of transition from 'tag' to the tokens in the corpus
                    if(prevtag not in self.transitions):
                        # intitialize to store count of transition from 'prevtag' to current tag
                    
                    #=======================TO-DO=======================
                    
                    # increment count for self.emissions
                    # increment count for self.transitions
                    # increment count for self.transitions_total
                    # increment count for self.emissions_total
                    

                # don't forget the stop probability for each sentence
                if prevtag not in self.transitions:
                    self.transitions[prevtag] = defaultdict(lambda: 0)
                
                #=======================TO-DO=======================
                # increment count for self.transitions from prevtag to final state
                # increment count for self.transitions_total for prevtag

    #=======================TO-DO=======================
    # calculate the transition probability prevtag -> tag
    def calculate_transition_prob(self, prevtag, tag):
        # TODO: Implement this. You can ignore smoothing in this task.
        return 0.0

    #=======================TO-DO=======================
    #calculate the probability of emitting token given tag
    def calculate_emission_prob(self, tag, token):
        # TODO: Implement this. You can ignore smoothing in this task.
        return 0.0

    # Write the model to an output file.
    # Doesn't need to be modified
    def writeResult(self):
        with open(self.OUTPUT_FILE, "w+") as f:
            for prevtag in self.transitions:
                for tag in self.transitions[prevtag]:
                    f.write("trans {} {} {}\n"
                        .format(prevtag, tag, self.calculate_transition_prob(prevtag, tag)))

            for tag in self.emissions:
                for token in self.emissions[tag]:
                    f.write("emit {} {} {}\n"
                        .format(tag, token, self.calculate_emission_prob(tag, token)))


if __name__ == "__main__":
    # Files
    TAG_FILE = sys.argv[1]
    TOKEN_FILE = sys.argv[2]
    OUTPUT_FILE = sys.argv[3]

    model = HMMTrain(TAG_FILE, TOKEN_FILE, OUTPUT_FILE)
    model.train()
    model.writeResult()