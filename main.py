# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import urllib

import matplotlib as matplotlib

import itertools
# %matplotlib inline
import pandas as pd
import numpy as np
import seaborn
from matplotlib import pyplot as plt
from IPython.display import display

import numpy as np
import pandas as pd
import re
import time
import io
from datasketch import MinHash, MinHashLSHForest
import openpyxl

from csv import reader


#Number of Permutations
permutations = 256

# Number of Recommendations to return
num_recommendations = 1


# def similarity_minhash(a: set, b: set) -> float:
def similarity_minhash(minhash_a, minhash_b) -> float:
    return sum([1 for a, b in zip(minhash_a, minhash_b) if a == b]) / len(minhash_a)
    '''
    sign_a = minhash_signature(a)
    sign_b = minhash_signature(b)
    return sum([1 for a, b in zip(sign_a, sign_b) if a == b]) / len(sign_a)
    '''


# Preprocess will split a string of text into individual tokens/shingles based on whitespace.
def preprocess(text):
    '''Remove all punctuation, lowercase all text, create unigram shingles (tokens) by
    separating any whitespace'''
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.lower()
    tokens = tokens.split()
    return tokens



def get_forest(data, perms):
    start_time = time.time()

    minhash = []

    for text in data['content']:
    #for text in data:
        tokens = preprocess(text)
        m = MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf8'))
        minhash.append(m)

    forest = MinHashLSHForest(num_perm=perms)

    for i, m in enumerate(minhash):
        forest.add(i, m)

    forest.index()

    print('It took %s seconds to build forest.' % (time.time() - start_time))

    return forest


def predict(text, database, perms, num_results, forest):
    start_time = time.time()

    tokens = preprocess(text)
    m = MinHash(num_perm=perms)
    for s in tokens:
        m.update(s.encode('utf8'))

    idx_array = np.array(forest.query(m, num_results))
    if len(idx_array) == 0:
        return None  # if your query is empty, return none

    result_content = database.iloc[idx_array]['content']

    # Note: minhash for query webpage is m

    # get tokens for top k
    top_k_tokens = preprocess(result_content.iloc[0])
    # create minhash for top k match
    top_k_m = MinHash(num_perm=perms)
    for s in top_k_tokens:
        top_k_m.update(s.encode('utf8'))

    print('It took %s seconds to query forest.' % (time.time() - start_time))

    result = m.jaccard(top_k_m)

    return result


# Press the green button in the gutter to run the script.
if __name__ == '__main__':



   #with open('C:\\Users\\ChristopherStinson\\OneDrive - AbleDocs Inc\\ADScan WebCrawler Near Duplicate Detection\\URL lists\\visited_urls.xlsx', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        #csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        #list_of_urls = list(csv_reader)
    df = pd.read_excel('C:\\Users\\ChristopherStinson\\OneDrive - AbleDocs Inc\\ADScan WebCrawler Near Duplicate Detection\\URL lists\\visited_urls.xlsx', header=None)

#list_of_urls.append("https://calendar.countyofrenfrew.on.ca/default/Week?StartDate=04/28/2019")
#list_of_urls.append("https://calendar.countyofrenfrew.on.ca/default/Week?StartDate=05/05/2019")
#list_of_urls.append("https://www.abledocs.com/en?gclid=EAIaIQobChMI1baaoK_H8gIVR-DICh1xhgQPEAAYASAAEgKoofD_BwE")

webpage_content_list = list()


#TODO need to create csv with smaller number of rows
for url in df.iterrows():
    print("Processing url:", url)
    try:
        webpage = urllib.request.urlopen(url[1].iloc[0])
        # type is bytes
        webpage_content = webpage.read()
        wbc_decoded = webpage_content.decode("utf-8")
        webpage_content_list.append(wbc_decoded)
    except :
        print("Error with url:", url)


d = {'url': df.values.tolist(), 'content': webpage_content_list}
db = pd.DataFrame(d)

forest = get_forest(db, permutations)



#test_webpage = urllib.request.urlopen('https://calendar.countyofrenfrew.on.ca/default/Week?StartDate=05/12/2019')
#test_webpage = urllib.request.urlopen('https://calendar.countyofrenfrew.on.ca/default/Month?StartDate=05/01/2019')
test_webpage = urllib.request.urlopen('https://www.bbc.com/')
test_webpage_content = test_webpage.read()
test_wbc_decoded = test_webpage_content.decode("utf-8")

result = predict(test_wbc_decoded, db, permutations, num_recommendations, forest)
print('\n Top Recommendation(s) is(are) \n', result)


'''TODO: is there a way to take a subset of the webpage content, ex: links on the page'''



'''
similar:
https://calendar.countyofrenfrew.on.ca/default/Week?StartDate=04/28/2019
https://calendar.countyofrenfrew.on.ca/default/Week?StartDate=05/05/2019

different:
https://www.abledocs.com/en?gclid=EAIaIQobChMI1baaoK_H8gIVR-DICh1xhgQPEAAYASAAEgKoofD_BwE

'''