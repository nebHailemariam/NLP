import math
import sys
import time
import numpy as np

from collections import defaultdict

# Magic strings and numbers
HMM_FILE = sys.argv[1]
TEXT_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]
TRANSITION_TAG = "trans"
EMISSION_TAG = "emit"
OOV_WORD = "OOV"  # check that the HMM file uses this same string
INIT_STATE = "init"  # check that the HMM file uses this same string
FINAL_STATE = "final"  # check that the HMM file uses this same string

"""
    The following code was adopted from
    https://github.com/melanietosik/viterbi-pos-tagger/blob/master/scripts/viterbi.py
"""


class Viterbi:
    def __init__(self):
        # transition and emission probabilities. Remember that we're not dealing with smoothing
        # here. So for the probability of transition and emission of tokens/tags that we haven't
        # seen in the training set, we ignore them by setting the probability an impossible value
        # of 1.0 (1.0 is impossible because we're in log space)

        self.transition = defaultdict(lambda: defaultdict(lambda: -float("inf")))
        self.emission = defaultdict(lambda: defaultdict(lambda: -float("inf")))
        # keep track of states to iterate over
        self.states = set()
        self.POSStates = set()
        # store vocab to check for OOV words
        self.vocab = set()

        # text to run viterbi with
        self.text_file_lines = []
        with open(TEXT_FILE, "r") as f:
            self.text_file_lines = f.readlines()

    def readModel(self):
        # Read HMM transition and emission probabilities
        # Probabilities are converted into LOG SPACE!
        with open(HMM_FILE, "r") as f:
            for line in f:
                line = line.split()

                # Read transition
                # Example line: trans NN NNPS 9.026968067100463e-05
                # Read in states as prev_state -> state
                if line[0] == TRANSITION_TAG:
                    (prev_state, state, trans_prob) = line[1:4]
                    self.transition[prev_state][state] = math.log(float(trans_prob))
                    self.states.add(prev_state)
                    self.states.add(state)

                # Read in states as state -> word
                elif line[0] == EMISSION_TAG:
                    (state, word, emit_prob) = line[1:4]
                    self.emission[state][word] = math.log(float(emit_prob))
                    self.states.add(state)
                    self.vocab.add(word)

        # Keep track of the non-initial and non-final states
        self.POSStates = self.states.copy()
        self.POSStates.remove(INIT_STATE)
        self.POSStates.remove(FINAL_STATE)
        self.POSStates = list(self.POSStates)

    # run Viterbi algorithm and write the output to the output file
    def runViterbi(self):
        result = []
        for line in self.text_file_lines:
            result.append(self.viterbiLine(line))

        # Print output to file
        with open(OUTPUT_FILE, "w") as f:
            for line in result:
                f.write(line)
                f.write("\n")

    # =======================TO-DO=======================
    def viterbiLine(self, line):
        words = line.split()
        for i, word in enumerate(words):
            # replace unseen words as oov
            if word not in self.vocab:
                words[i] = "OOV"

        # Initialize
        viterbi, back_pointer = self.initialize(words)

        # Iterate
        self.forward(words, viterbi, back_pointer)

        # Obtain optimal sequence
        return self.optimal_sequence(words, viterbi, back_pointer)

    def initialize(self, words) -> list:
        viterbi = []
        back_pointer = []

        # Create a path probability matrix and viterbi matrix
        for i in range(len(self.POSStates)):
            viterbi.append([0] * len(words))
            back_pointer.append([None] * len(words))

        # Initialize the starting probabilities
        for state_index in range(len(self.POSStates)):
            viterbi[state_index][0] = (
                self.transition["init"][self.POSStates[state_index]]
                + self.emission[self.POSStates[state_index]][words[0]]
            )
            back_pointer[state_index][0] = 0

        return viterbi, back_pointer

    def forward(self, words, viterbi, back_pointer) -> list:
        for word_index in range(1, len(words)):
            for state_index in range(len(self.POSStates)):
                best_prob = -float("inf")
                best_path = None

                for prev_state_index in range(len(self.POSStates)):
                    prob = (
                        viterbi[prev_state_index][word_index - 1]
                        + self.transition[self.POSStates[prev_state_index]][
                            self.POSStates[state_index]
                        ]
                        + self.emission[self.POSStates[state_index]][words[word_index]]
                    )

                    if prob > best_prob:
                        best_prob = prob
                        best_path = prev_state_index

                viterbi[state_index][word_index] = best_prob
                back_pointer[state_index][word_index] = best_path
        return viterbi, back_pointer

    def optimal_sequence(self, words, viterbi, back_pointer) -> list:
        try:
            # An array containing the optimal sequence of states
            optimal_sequence = []
            optimal_index = []
            # Initialize the optimal sequence
            for _ in range(len(words)):
                optimal_sequence.append(None)
                optimal_index.append(None)

            # Find the optimal state for the last word
            arg_max = viterbi[0][len(words) - 1]
            for state_index in range(1, len(self.POSStates)):
                if viterbi[state_index][len(words) - 1] > arg_max:
                    arg_max = viterbi[state_index][len(words) - 1]
                    optimal_index[len(words) - 1] = state_index

            # Set the optimal index for the last word
            optimal_sequence[len(words) - 1] = self.POSStates[
                optimal_index[len(words) - 1]
            ]

            # Iteratively backtrack to find the optimal sequence
            for i in range(len(words) - 1, 0, -1):
                optimal_index[i - 1] = back_pointer[optimal_index[i]][i]
                optimal_sequence[i - 1] = self.POSStates[optimal_index[i - 1]]
        except:
            return ""
        return " ".join(optimal_sequence)


if __name__ == "__main__":
    # Mark start time
    t0 = time.time()
    viterbi = Viterbi()
    viterbi.readModel()
    viterbi.runViterbi()
    # Mark end time
    t1 = time.time()
    print("Time taken to run: {}".format(t1 - t0))
