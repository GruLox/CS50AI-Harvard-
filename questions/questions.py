import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
            files[filename] = content
    
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.tokenize.word_tokenize(document)
    stopwords = nltk.corpus.stopwords.words("english")

    words = [
        word.lower() for word in words if word.lower() not in string.punctuation and word.lower() not in stopwords
    ]

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    totalDocuments = len(documents)
    idfs = {}
    for key in documents:
        words = documents[key]
        for word in words:
            documentsThatContain = len([key for key in documents if word in documents[key]])
            idf = math.log(totalDocuments / documentsThatContain)
            idfs[word] = idf
    
    return idfs



def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    values = { filename: 0 for filename in files }
    for file in files:
        for word in query:
            if word in files[file]:
                tf = files[file].count(word)
                tfidf = tf * idfs[word]
                values[file] += tfidf
    
    sortedItems = sorted(values.items(), key=lambda item: item[1], reverse=True)

    sortedFiles = [item[0] for item in sortedItems]

    return sortedFiles[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    values = {sentence: 0 for sentence in sentences}
    queryTermDensity = {}

    for sentence in sentences:
        for word in query:
            if word in sentences[sentence]:
                values[sentence] += idfs[word]

        sentenceWords = sentences[sentence]
        totalWords = len(sentenceWords)
        queryWordsInSentence = len([word for word in sentenceWords if word in query])
        queryTermDensity[sentence] = queryWordsInSentence / totalWords

    sortedItems = sorted(values.items(), key=lambda item: (item[1], queryTermDensity[item[0]]), reverse=True)

    sortedSentences = [item[0] for item in sortedItems]

    return sortedSentences[:n]


if __name__ == "__main__":
    main()
