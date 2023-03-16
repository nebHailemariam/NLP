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

        # Initialize DP matrix for Viterbi here (we suggest using a dictionary)
        # You will require a path probability matrix and a backpointer matrix

        for i, word in enumerate(words):
            # replace unseen words as oov
            if word not in self.vocab:
                words[i] = "OOV"

        O = self.vocab  # Observation space
        S = list(self.POSStates)  # State space
        Y = words  # Sequence of observations
        A = self.transition  # Transition matrix
        B = self.emission  # Emission matrix

        N = len(O)
        K = len(S)
        lookup = {}

        for i, word in enumerate(O):
            lookup[word] = i
        T = len(Y)
        T1 = [[0] * T for i in range(K)]
        T2 = [[None] * T for i in range(K)]

        # Predicted tags
        X = [None] * T
        """
        Initialize start probabilities
        """
        for i in range(K):
            T1[i][0] = A["init"][S[i]] + B[S[i]][Y[0]]
            T2[i][0] = 0

        """
        Forward step
        """
        for i in range(1, T):
            for j in range(K):
                best_prob = float("-inf")
                best_path = None

                for k in range(K):
                    prob = T1[k][i - 1] + A[S[k]][S[j]] + B[S[j]][Y[i]]

                    if prob > best_prob:
                        best_prob = prob
                        best_path = k

                T1[j][i] = best_prob
                T2[j][i] = best_path

        """
        Backward step
        """
        try:
            z = [None] * T
            argmax = T1[0][T - 1]

            for k in range(1, K):
                if T1[k][T - 1] > argmax:
                    argmax = T1[k][T - 1]
                    z[T - 1] = k

            X[T - 1] = S[z[T - 1]]
            for i in range(T - 1, 0, -1):
                z[i - 1] = T2[z[i]][i]
                X[i - 1] = S[z[i - 1]]
        except:
            return ""

        return " ".join(X)


if __name__ == "__main__":
    # Mark start time
    t0 = time.time()
    viterbi = Viterbi()
    viterbi.readModel()
    viterbi.runViterbi()
    # Mark end time
    t1 = time.time()
    print("Time taken to run: {}".format(t1 - t0))
