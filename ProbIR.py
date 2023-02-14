import csv
import re

from nltk import PorterStemmer
from tqdm import tqdm 
import numpy as np
import scipy as sp # for sparse arrays
from time import sleep
import matplotlib.pyplot as plt 

class Document:
    """Generic class to describe a document, under the assuption that each document only has a title and a text."""
    def __init__(self,title, text):
        self.title = title
        self.text = text
    def __repr__(self):
        return self.title

    def get_text(self):
        return self.text
    
    def to_int(self):
        return int(self.title)

def normalize(text, stemmer=False):
    """String normalization with additional stemmer. The function removes punctuation and gets all the character
        of the string to be lowercase. Stemmer, which bt default is off, passes the obtained string to the Porter Stemmer.
        The function returns a list of strings."""
    normalized = re.sub(r'[^\w^\s^-]','',text).lower()
    if(stemmer):
        stemmer = PorterStemmer()
        normalized = stemmer.stem(normalized)
    
    return list(normalized.split())

def make_dict(corpus):
    """Given a list of Documents (corpus), the functions returns the dictionary of the corpus."""
    dictionary = []
    dict_set = set()

    for doc in corpus:
        for word in doc:
            if(word in dict_set):
                continue
            else:
                dictionary.append(word)
                dict_set.add(word)
    return dictionary

def inverted_index(corpus,stemmer=False):
    """This function builds an inverse index based on a list of Documents,
        returning also the tf and idf for each word in two separate lists.
        function also allows for the use of a stemmer (by default it is off)."""

    corpus_norm = [normalize(text.get_text(), stemmer) for text in corpus]
    dictionary = make_dict(corpus_norm)
    inv_idx = {}
    n= len(corpus_norm)
    for token in dictionary:
        inv_idx[token] = []
    
    for i in tqdm(range(len(corpus_norm)), desc= "Computing reverse index"):
        for token in corpus_norm[i]:
            inv_idx[token].append(i)

    tf = {}
    for key in tqdm(inv_idx.keys(), desc="Computing tf scores"):
        curr = 0
        count = np.zeros(n,dtype = int)
        doc_in_word = len(inv_idx[key])
        for i in range(doc_in_word):
            if(inv_idx[key][i]==curr):
                count[curr] += 1
            else:
                curr=inv_idx[key][i]
                count[curr]+=1
        tf[key] = sp.sparse.csr_matrix(count)
        inv_idx[key] = [*set(inv_idx[key])]
    
    idf={}
    for key in inv_idx.keys():
        idf[key]=np.log(n/len(inv_idx[key]))
    return inv_idx,tf,idf

def count_relevant(relevant, idx):
    """Return the number of relevant documents which contain the word indexed by "idx"."""
    idx_set = set(idx)
    count = 0
    for element in relevant:
        if(element in idx_set):
            count+=1
    return int(count)

def ordered_print(list):
    """Dummy print function needed to print the retrieved documents together with their position."""
    for i in range(len(list)):
        print("{}. {}".format(i+1, list[i]), flush=True)

class ProbIR:
    """Class to create an IR system based on the Okapi BM25 model."""
    def __init__(self,corpus,index,tf,idf):
        self.corpus = corpus
        self.idx = index
        self.tf = tf
        self.idf = idf
    
    @classmethod
    def from_corpus(cls, corpus, stemmer = False):
        """Initialize ProbIR computing automatically the needed objects from a corpus."""
        idx, tf, idf = inverted_index(corpus,stemmer)
        return cls(corpus,idx,tf,idf)

    def rsv_scores(self, query,b,k,k2 = None, relevant = None, nonrelevant = None, test_mode = False):
        """Compute the RSV scores of the documents given a query. Relevance feedback is also available
            passing a list of relevant books and nonrelevant ones from the query."""
        avg = 0
        for i in range(len(self.corpus)):
            avg += len(self.corpus[i].get_text())
        avg /= len(self.corpus)

        scores = np.zeros(len(self.corpus))
        # the relevant list is used to switch between the vanilla score function and the one
        # derived from user feedback, involving relevant and non relevant documents.
        # Note: if a word in the query is not present inside the dictionary it will be ignored.

        if(relevant == None):
            for word in query:
                try:
                    for d in range(len(self.corpus)):
                        l_d = len(self.corpus[d].get_text())
                        tf_td = self.tf[word].todense()[0,d]
                        num = (k+1)*tf_td
                        den = k*((1-b)+b*(l_d/avg)) + tf_td
                        if(k2 != None):
                            opt = ((k2+1)*tf_td)/(k2+tf_td)
                        else: 
                            opt = 1
                        
                        scores[d] += self.idf[word]*(num/den)*opt
                except KeyError:
                    if(test_mode == False):
                        print("Word {} not found, it will be ignored.".format(word))
                    continue
        else:
            vr = len(relevant)
            n = len(self.corpus)
            for word in query:
                try:
                    vr_t = count_relevant(relevant, self.idx[word])
                    vnr_t = count_relevant(nonrelevant, self.idx[word])
                    df_t = np.exp(self.idf[word])/n
                    for d in range(len(self.corpus)):
                        l_d = len(self.corpus[d].get_text())
                        tf_td = self.tf[word].todense()[0,d]
                        num = (k+1)*tf_td
                        den = np.abs(k*((1-b)+b*(l_d/avg)) + tf_td)
                        if(den == 0):
                            den = 1e-4 # added to avoid zero division
                        relevance_num = (vr_t + .5) /(vnr_t + .5)
                        relevance_den = np.abs((df_t - vr_t + .5)/(n - df_t - vr + vr_t + .5))
                        if(relevance_den == 0):
                            relevance_den = 1e-4 # added to avoid zero division

                        scores[d] += np.log(1e-4 + (relevance_num/(relevance_den))*(num/(den)))
                except KeyError:
                    if(test_mode == False):
                        print("Word {} not found, it will be ignored.".format(word))
                    continue
        
        return scores

    def query_relevance(self, query, relevant, nonrelevant, stemmer=False, results=5, b=.75, k=1.6, k2 = None,test_mode = False):
        """Internal method that implements the relevance scoring, iterates until the user is satisfied. """
        
        # TODO: when re-called, pass also previous documents
        scores = self.rsv_scores(query, b, k, k2, relevant, nonrelevant,test_mode)

        idx_sorted = scores.argsort()[::-1]
        sel_books = [self.corpus[i] for i in idx_sorted[:results]]
        ordered_print(sel_books)
        sleep(1)
        ans = input("Are you satisfied with the obtained results? (y/n) ")
        if(ans == "y"):
            return sel_books
        elif(ans == "n"):
            print("\nTo help us refine the results, pick the relevant results \n(enter the number of the document)")
            flag = True
            while (flag):
                rel_doc = input()
                if(rel_doc == ""):
                    flag = False
                else:
                    relevant.append(idx_sorted[int(rel_doc)-1])

            nonrelevant+= list(idx_sorted[:results])
            for elem in relevant:
                nonrelevant.remove(elem)
            if(test_mode==False):
                    print("Retrieving new documents...")
            sel_books = self.query_relevance(query, relevant, nonrelevant, stemmer, results, b, k, k2)
            return sel_books
    
    def query(self, query, stemmer = False, results = 5, b = .75, k = 1.6, k2 = None, pseudorel = 0, test_mode = False):
        """Submit a query to the IR system, with optional relevance feedback

            
            Parameters
            ----------
            query: str
                The query to be submitted to the IR system.
            stemmer: bool
                Allows the query to be passed through a stemmer. Needed only if the dictionary of the corpus was computed using a stemmer.
                By default it is off.
            results : int >= 1
                Number of documents to show from the query results.
                Default value is 5.
            b : double in [0,1]
                Parameter that regulates the strenght of the normalization with respect to the length, 
                should be between 0 (no normalization) to 1 (full scaling).
                Default value is 0.75.
            k : double >= 0
                Parameter that regulates the document term frequency scaling.
                For k=0 we have a standard binary model, while suggested values range from 1.2 to 2.
                Default value is 1.6.
            k2 : double >=0, None
                Additional parameter that regulates calibrates term frequency scaling of the query. This is an additional parameter
                only useful for very long queries. Suggested values range from 1.2 to 2.
                By default, the parameter is bypassed.
            pseudorel : int >= 0
                Number of documents of the pseudo-relevance feedback.
                By default the parameter is set to 0 (no relevance feedback).
            test_mode : bool
                Parameters that acts like a "verbose" switch, if turned off will omit all the prints and return the selected documents,
                not allowing feedback user for relevance of the documents.
                Since it is only used for evaluation of the system purposes, by default it is set to False."""
        
        query = normalize(query,stemmer)
        if(test_mode == False):
            print(query)
        scores = self.rsv_scores(query,b,k,k2 = k2, test_mode = test_mode)
        idx_sorted = scores.argsort()[::-1]
        sel_books = [self.corpus[i] for i in idx_sorted[:results]]
        
        # a positive pseudorel argument will activate the following scope, allowing pseudo-relevance feedback.
        # In the case where the pseudorel argument is greater then the 
        if(pseudorel > 0):
            if(test_mode == False):
                print("Computing pseudo-relevance feedback using first {} documents.".format(pseudorel))
            relevant = []
            for i in range(pseudorel):
                relevant.append(idx_sorted[i])
            nonrelevant = []
            sel_books = self.__pseudorel_qry(query, relevant, nonrelevant, stemmer,results, b, k,k2, test_mode)
        if(test_mode):
            return sel_books
        
        ordered_print(sel_books)
        sleep(1) # needed for sinchronization between print of the returned documents and user feedback 
        ans = input("Are you satisfied with the obtained results? (y/n) ")
        if(ans == "y"):
            return sel_books
        elif(ans == "n"):
            print("\nTo help us refine the results, pick the relevant results \n(enter the number of the document)")
            flag = True
            relevant = []
            while (flag):
                rel_doc = input()
                if(rel_doc == ""):
                    flag = False
                else:
                    relevant.append(idx_sorted[int(rel_doc)-1])
            # documents that were seen by the user but not marked as relevant are 
            # automatically interpreted as non-relevant.
            nonrelevant = list(idx_sorted[:results])
            for elem in relevant:
                nonrelevant.remove(elem)
                if(test_mode==False):
                    print("Retrieving new documents...")
            sel_books = self.query_relevance(query, relevant, nonrelevant, stemmer,results, b, k, k2, test_mode)
            return sel_books
        else:
            return sel_books
    
    
    def __pseudorel_qry(self, query, relevant, nonrelevant, stemmer, results, b, k, k2, test_mode):
        """Internal method that implements the pseudo-relevance feedback."""
        scores = self.rsv_scores(query, b, k, k2, relevant, nonrelevant, test_mode)
        idx_sorted = scores.argsort()[::-1]
        sel_books = [self.corpus[i] for i in idx_sorted[:results]]
        return sel_books

    