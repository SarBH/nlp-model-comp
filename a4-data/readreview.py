# import xml.etree.ElementTree as ET
from lxml import etree
# from bs4 import BeautifulSoup


def extract_text_from_reviews(path_to_file, label):
    reviews = []
    labels = []
    
#     infile = open(path_to_file, "r")
#     contents = infile.read()
#     soup = BeautifulSoup(contents, 'xml')
    root = etree.parse(path_to_file)

    for review_tag in root.findall('review'):
        print(review_tag) 
        text = review_tag.find('review_text')
        print(text.text)
        reviews.append(text)
        labels.append(label)
    
    return reviews, labels
        
    
def embed_text(text_list, embeddings_dict):
    x_all = []
    for sample_idx, sample in enumerate(text_list):
        # 1. remove punctuations
        paragraph = paragraph.translate(str.maketrans('','',string.punctuation))
        paragraph = paragraph.replace('\n', ' ')
        
        # 2. tokenize
        tokens = nltk.word_tokenize(paragraph)
        
        # 3. remove stop words
        vectors = []
        for token in tokens:
            if not token in stop_words:
                try:
                    vector = embedding[token.lower()]
                    vectors.append(vector)
                except KeyError:
                    continue
        x_all.append(vectors)
    return x_all

if __name__ == "__main__":
    filepaths_dict = {'1': './a4-data/q2/positive.review.xml',
                 '0': './a4-data/q2/negative.review.xml'}

    x_all, all_labels = [], []
    for label, filepath in filepaths_dict.items():
        reviews, labels = extract_text_from_reviews(filepath, label)
        
        embedded_reviews = embed_text(reviews, embeddings_dict)
        
        x_all.append(embedded_reviews)
        all_labels.append(labels)
