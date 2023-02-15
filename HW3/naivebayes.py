# -*- coding: utf-8 -*-
from collections import defaultdict, Counter
from operator import itemgetter
from math import log
from typing import DefaultDict
import sys
import io

# Define A NaiveBayes class which is used to distinguish between languages using a character bigram model.

# Do create necessary variables wherever required. You might have to add self in some definitions and calls and keep variables in self as well

class NaiveBayes():

    def extract_ngrams(self,x: str, n=2) -> "list[str]":
        """
        Train a Naive-Bayes model

        :param x: The document which needs to be decomposed into ngrams.
        :para n: the order of ngrams.
        :return: list of ngrams
        """
        ###TODO###
        #extract character ngrams
        x = list(x)
        n_grams = []
        
        if "\n" in x:
            n_grams.append("\n")
        for i in range(0, len(x) - n, n):
            n_grams.append(x[i : i + n])

        return n_grams

    def smoothed_log_likelihood(self, w: str, c: str, k: int, count: 'DefaultDict[str, Counter]', vocab: "set[str]") -> float:
        """
        :param w: word in the vocab
        :para c: class label
        :param k: The value added to the numerator and denominator to smooth likelihoods
        :para count: Dictionary containing the count of label c occuring with an ngram w
        :param vocab: the vocabulary for the model
        :type b: int
        :return: the log likelihood value after smoothening
        """
        ###TODO###
        #apply smoothing
        word_count = 0
        vocabulary_count = {}

        for doc in count[c]:
            word_count += doc.count(w)
            
            for vocabulary in vocab:
                if vocabulary != w:
                    if vocabulary in vocabulary_count:
                        vocabulary_count[vocabulary] += doc.count(vocabulary)
                    else:
                        vocabulary_count[vocabulary] = doc.count(vocabulary) + k
        
        
        summation_of_vocab_count = 0

        for vocab in vocabulary_count.keys():
            summation_of_vocab_count += vocabulary_count[vocab]

        log_likelihood = log((word_count + k)/(summation_of_vocab_count + word_count + k))

        return log_likelihood

    def train_nb(self, docs: "list[tuple[str, str]]", k: int = 1, n: int = 2) -> "tuple[dict[str, float], DefaultDict[str, DefaultDict[str, float]], set[str], set[str]]":
        ###TODO###
        """
        Train a Naive-Bayes model

        :param docs: The documents, each associated with a clas label (document, label)
        :type docs: list[tuple[str, str]]
        :param k: The value added to the numerator and denominator to smooth likelihoods
        :type k: int
        :para n: the order of ngrams.
        :type b: int
        :return: The log priors, log likelihoods, the classes, and the vocabulary for the model at a tuple
        :rtype: tuple[dict[str, float], DefaultDict[str, DefaultDict[str, float]], set[str], set[str]]
        """
        num_of_docs = len(docs)
        log_priors = {}

        for doc in docs:
            doc_label = doc[0]

            if doc_label in log_priors:
                log_priors[doc_label] += 1
            else:
                log_priors[doc_label] = 1
        
        labels = log_priors.keys()

        for label in labels:
            log_priors[label] = log(log_priors[label]/num_of_docs)

        vocabulary = {}
        for doc in docs:
            for vocab in self.extract_ngrams(doc[1]):
                for v in vocab:
                    vocabulary[v] = True

        vocabulary = vocabulary.keys()
        print(vocabulary)
        big_doc = {}

        for label in labels:
            big_doc[label] = []

        for doc in docs:
            big_doc[doc[0]].append(doc[1])
        
        log_likelihoods = {}
        for word in vocabulary:
            for label in labels:
                log_likelihood = self.smoothed_log_likelihood(word, label, k, big_doc, vocabulary)
                
                if word not in log_likelihoods:
                    log_likelihoods[word] = {label:log_likelihood}
                else:
                    log_likelihoods[word][label] = log_likelihood
            
        return log_priors, log_likelihoods, labels, vocabulary

    def classify(self, testdoc: str, log_prior: "dict[str, float]", log_likelihood: "DefaultDict[str, DefaultDict[str, float]]", classes: "set[str]", vocab: "set[str]", k: int=1, n: int=2) -> str:
        ###TODO###
        """Given a trained NB model (log_prior, log_likelihood, classes, and vocab), returns the most likely label for the input document.

        :param textdoc str: The test document.
        :param log_prior dict[str, float]: The log priors of each category. Categories are keys and log priors are values.
        :param log_likelihood DefaultDict[str, DefaultDict[str, float]]: The log likelihoods for each combination of word/ngram and class.
        :param classes set[str]: The set of class labels (as strings).
        :param vocab set[str]: The set of words/negrams in the vocabulary.
        :param k int: the value added in smoothing.
        "param n int: the order of ngrams.
        :return: The best label for `testdoc` in light of the model.
        :rtype: str
        """

        ##TODO
        # Extract a set of ngrams from `testdoc`
        doc = self.extract_ngrams(testdoc)
        ##TODO
        # Initialize the sums for each class. These will be the "scores" based on which class will be assigned.
        class_sum = {}
        ##TODO
        # Iterate over the classes, computing `class_sum` for each
        for c in classes:
            ##TODO
            # Initialize `class_sum` with the log prior for the class
            ##TODO
            # Then add the likelihood for each in-vocabulary word/ngram in the document
            class_sum[c] = 0

            for vocab in self.extract_ngrams(testdoc):
                for v in vocab:
                    try:
                        class_sum[c] += log_likelihood[v][c]
                    except:    
                        v_count = testdoc.count(v) + k
                        v_complement_count = 0

                        for v_complement in set(testdoc):
                            v_complement_count += testdoc.count(v_complement) + k
                        
                        class_sum[c] += log(v_count/v_complement_count)
        best_class = list(class_sum.keys())[0]
        best_class_value = class_sum[best_class]

        for a_class in class_sum.keys():
            if best_class_value <= class_sum[a_class]:
                best_class = a_class
                best_class_value = class_sum[a_class]
        
        return best_class
        
            
        

    def precision(self,tp: "dict[str, int]", fp: "dict[str, int]") -> float:
        return tp / (tp + fp)

    def recall(self,tp: "dict[str, int]", fn: "dict[str, int]") -> float:
        return tp / (tp + fn)

    def micro_precision(self, tp: "dict[str, int]", fp: "dict[str, int]") -> float:
        tp_sum = sum(tp.values())
        fp_sum = sum(fp.values())
        return tp_sum / (tp_sum + fp_sum)

    def micro_recall(self, tp: "dict[str, int]", fn: "dict[str, int]") -> float:
        tp_sum = sum(tp.values())
        fn_sum = sum(fn.values())
        return tp_sum / (tp_sum + fn_sum)

    def micro_f1(self, tp: "dict[str, int]", fp: "dict[str, int]", fn: "dict[str, int]") -> float:
        mp = self.micro_precision(tp, fp)
        mr = self.micro_recall(tp, fn)
        return 2 * (mp * mr) / (mp + mr)

    def macro_precision(self, tp: "dict[str, int]", fp: "dict[str, int]") -> float:
        n = len(tp)
        return (1 / n) * sum([self.precision(tp[c], fp[c]) for c in tp.keys()])

    def macro_recall(self, tp: "dict[str, int]", fn: "dict[str, int]") -> float:
        n = len(tp)
        return (1 / n) * sum([self.recall(tp[c], fn[c]) for c in tp.keys()])

    def macro_f1(self, tp: "dict[str, int]", fp: "dict[str, int]", fn: "dict[str, int]") -> float:
        n = len(tp)
        return 2 * (self.macro_precision(tp, fp) * self.macro_recall(tp, fn)) / (self.macro_precision(tp, fp) + self.macro_recall(tp, fn))

    def evaluate(self, train: "list[tuple[str, str]]", eval: "list[tuple[str, str]]", n: int=2):
        log_prior, log_likelihood, classes, vocab = self.train_nb(train, n = n)
        # Initialize dictionaries for true positives, false positives, and false negatives
        tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
        confusion = defaultdict(lambda: defaultdict(int))
        for c_ref, doc in eval:
            c_hyp = self.classify(doc, log_prior, log_likelihood, classes, vocab, n = n)
            confusion[c_ref][c_hyp] += 1
            if c_ref == c_hyp:
                tp[c_ref] += 1
            else:
                fn[c_ref] += 1
                fp[c_hyp] += 1

        print(f'Macro-averaged precision:\t{self.macro_precision(tp, fp)}')
        print(f'Macro-averaged recall:\t{self.macro_recall(tp, fn)}')
        print(f'Macro-averaged F1:\t{self.macro_f1(tp, fp, fn)}')
        print(f'Micro-averaged precision:\t{self.micro_precision(tp, fp)}')
        print(f'Micro-averaged recall:\t{self.micro_recall(tp, fn)}')
        print(f'Micro-averaged F1:\t{self.micro_f1(tp, fp, fn)}')

        return self.macro_precision(tp, fp), self.macro_recall(tp, fn), self.macro_f1(tp, fp, fn), self.micro_precision(tp, fp), self.micro_recall(tp, fn), self.micro_f1(tp, fp, fn)


"""
The following code is used only on your local machine. The autograder will only use the functions in the NaiveBayes class.            

You are allowed to modify the code below. But your modifications will not be used on the autograder.
"""
if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Usage: python3 naivebayes.py TRAIN_FILE_NAME TEST_FILE_NAME")
        sys.exit(1)

    train_txt = sys.argv[1]
    test_txt = sys.argv[2]

    with open(train_txt, encoding='utf-8') as f:
        train = [tuple(l.split('\t')) for l in f]
    
    with open(test_txt, encoding='utf-8') as f:
        test = [tuple(l.split('\t')) for l in f]

    tmp=NaiveBayes()
    map, mar, maf, mp, mr, mf=tmp.evaluate(train, test, n=2)

