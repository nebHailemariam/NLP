import sys
import re

from collections import defaultdict


class HMMTrainSmoothed:
    def __init__(self, TAG_FILE, TOKEN_FILE, OUTPUT_FILE):
        self.TAG_FILE = TAG_FILE
        self.TOKEN_FILE = TOKEN_FILE
        self.OUTPUT_FILE = OUTPUT_FILE
        # Vocabulary
        self.vocab = {}
        self.OOV_WORD = "OOV"
        self.INIT_STATE = "init"
        self.FINAL_STATE = "final"
        # Transition and emission probabilities
        self.emissions = (
            {}
        )  # dictionary to store count of transition from POS tag state to tokens in the corpus
        self.transitions = (
            {}
        )  # dictionary to store count of transition from previous POS tag state to current state
        self.transitions_total = defaultdict(
            lambda: 0
        )  # total number of transitions from one state to all others
        self.emissions_total = defaultdict(
            lambda: 0
        )  # total number of transitions from one state to all tokens
        self.count_of_transitions = 0
        self.count_of_emissions = 0
        self.d = 0.75  # Discount factor

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

                for tag, token in pairs:
                    # this block is a little trick to help with out-of-vocabulary (OOV)
                    # words.  the first time we see *any* word token, we pretend it
                    # is an OOV.  this lets our model decide the rate at which new
                    # words of each POS-type should be expected (e.g., high for nouns,
                    # low for determiners).

                    if token not in self.vocab:
                        self.vocab[token] = 1
                        token = self.OOV_WORD

                    # =======================TO-DO=======================

                    if tag not in self.emissions:
                        pass
                        # initialize to store count of transition from 'tag' to the tokens in the corpus
                    if prevtag not in self.transitions:
                        # intitialize to store count of transition from 'prevtag' to current tag
                        pass
                    # =======================TO-DO=======================

                    # increment count for self.emissions
                    # increment count for self.transitions
                    # increment count for self.transitions_total
                    # increment count for self.emissions_total

                # don't forget the stop probability for each sentence
                if prevtag not in self.transitions:
                    self.transitions[prevtag] = defaultdict(lambda: 0)

                # =======================TO-DO=======================
                # increment count for self.transitions from prevtag to final state
                # increment count for self.transitions_total for prevtag
                # Count the total number of unique transitions for count_of_transitions
                # Count the total number of unique emissions for count_of_emissions

    # =======================TO-DO=======================
    # calculate the transition probability prevtag -> tag
    def calculate_transition_prob(self, prevtag, tag):
        # Discount factor is defined in init

        # Calculate the continuation probability using the 'novel continuation for a tag' (number of word types seen to precede tag) and the total number of bigram types
        # You can choose to fill the template for the function calculate_novel_continuation and use it or you can do write your own code here.

        # Calculate the backoff probability using the count of the transitions with prevtag and the number of unique words than can follow prevtag

        # Calculate the smoothed probability

        return 0.0

    # =======================TO-DO=======================
    # calculate the probability of emitting token given tag
    def calculate_emission_prob(self, tag, token):
        # Calculate the continuation probability using the 'novel continuation for a token' (number of tags pointing to token) and the total number of emissions
        # You can choose to fill the template for the function calculate_novel_continuation and use it or you can do write your own code here.

        # Calculate the backoff probability using the count of the emissions from tag and the number of unique tokens than can be emitted from tags

        # Calculate the smoothed probability

        return 0.0

    def calculate_novel_continuation(self, tag_or_token, transition=True):
        # If transition is true, this function was called by calculate_transition_prob otherwise calculate_emission_prob
        main_checker = self.transitions
        if transition == False:
            main_checker = self.emissions

        # Loop on main_checker to either:
        # 1. For transitions, get the number of all prevtags which have tag as a transition
        # or, 2. For emission, get the number of all tags which have token emitting from it and return it.

        return 0.0

    # Write the model to an output file.
    # Doesn't need to be modified
    def writeResult(self):
        with open(self.OUTPUT_FILE, "w+") as f:
            for prevtag in self.transitions:
                for tag in self.transitions[prevtag]:
                    f.write(
                        "trans {} {} {}\n".format(
                            prevtag, tag, self.calculate_transition_prob(prevtag, tag)
                        )
                    )

            for tag in self.emissions:
                for token in self.emissions[tag]:
                    f.write(
                        "emit {} {} {}\n".format(
                            tag, token, self.calculate_emission_prob(tag, token)
                        )
                    )


if __name__ == "__main__":
    # Files
    TAG_FILE = sys.argv[1]
    TOKEN_FILE = sys.argv[2]
    OUTPUT_FILE = sys.argv[3]

    model = HMMTrainSmoothed(TAG_FILE, TOKEN_FILE, OUTPUT_FILE)
    model.train()
    model.writeResult()
