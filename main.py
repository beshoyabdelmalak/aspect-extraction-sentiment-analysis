import os
import re

import tensorflow as tf
from elasticsearch import Elasticsearch
from nltk.tokenize import sent_tokenize

from aspect_extractor.model.aspect_model import ASPECTModel
from aspect_extractor.model.config import Config
from aspect_extractor.evaluate import align_data
import sentiment_analyzer.code.sa_aspect_oop as sa
from aspect_list import ASPECTS

es = Elasticsearch()


def extract_aspect(model, sentence):
    words_raw = sentence.strip().split()
    if not words_raw:
        return None
    words = [model.config.processing_word(w) for w in words_raw]
    if type(words[0]) == tuple:
        words = zip(*words)
    labels_pred, sequence_lengths = model.predict_batch([words])
    preds = [model.idx_to_tag[idx] for idx in list(labels_pred[0])]

    result = dict(zip(words_raw, preds))
    aspect_found = False
    for word in result:
        if result[word] == "B-A":
            aspect_found = True

    if not aspect_found:
        return None
    return result


def clean_data(docs):
    outputProducts = []
    for hit in docs["hits"]["hits"]:
        item = dict()
        item["review"] = hit["_source"]["review"]
        item["asin"] = hit["_source"]["asin"]
        item["id"] = hit["_id"]
        outputProducts.append(item)
    return outputProducts


def pre_processing(review):
    processed_sentences = []
    sentences = sent_tokenize(review)
    for sentence in sentences:
        processed_sentence = re.sub(r"[^a-zA-Z0-9#_]", " ", str(sentence))
        # processed_sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_sentence)
        # processed_sentence = re.sub(r'\s+', ' ', processed_sentence, flags=re.I)
        processed_sentence = processed_sentence.lower()
        processed_sentences.append(processed_sentence)
    return processed_sentences


def main():
    # aspect extraction configurations
    config = Config()
    aspect_model = ASPECTModel(config)
    aspect_model.build()
    aspect_model.restore_session(config.dir_model)

    # sentiment analyzer configurations
    batch_size = 128
    nb_sentiment_label = 3
    nb_sentiment_for_word = 3
    embedding_size = 100
    nb_lstm_inside = 256
    layers = 1
    TRAINING_ITERATIONS = 2000
    LEARNING_RATE = 0.1
    WEIGHT_DECAY = 0.0001
    label_dict = {"aspositive": 1, "asneutral": 0, "asnegative": 2}
    data_dir = os.path.join(os.getcwd(), "sentiment_analyzer/data/ABSA_SemEval2014/")
    domain = "Laptops"
    seq_max_len = 42
    nb_linear_inside = 256

    flag_word2vec = False
    flag_addition_corpus = False
    flag_change_file_structure = False
    flag_use_sentiment_embedding = False
    flag_use_sentiment_for_word = False
    flag_train = False

    negative_weight = 1.0
    positive_weight = 1.0
    neutral_weight = 1.0

    sess = tf.Session()

    data = sa.Data(
        domain,
        data_dir,
        flag_word2vec,
        label_dict,
        seq_max_len,
        flag_addition_corpus,
        flag_change_file_structure,
        negative_weight,
        positive_weight,
        neutral_weight,
        flag_use_sentiment_embedding,
    )

    if flag_use_sentiment_for_word:
        embedding_size = embedding_size + nb_sentiment_for_word

    sentiment_model = sa.Model(
        batch_size,
        seq_max_len,
        nb_sentiment_label,
        nb_sentiment_for_word,
        embedding_size,
        nb_linear_inside,
        nb_lstm_inside,
        layers,
        TRAINING_ITERATIONS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        flag_train,
        flag_use_sentiment_for_word,
        sess,
    )

    sentiment_model.modeling()

    if sentiment_model.flag_train:
        sentiment_model.train(data)
    else:
        # get the two trained models and feed the data to them
        sentiment_model.load_model()
        # sentiment_model.evaluate(data, True, sentiment_model.flag_train)
        doc = {"size": 100, "query": {"match_all": {}}}
        # handle the first 10000 reviews
        all_reviews = es.search(index="reviews", body=doc, scroll="1m")
        cleaned_reviews = clean_data(all_reviews)
        handle_reviews(cleaned_reviews, data, aspect_model, sentiment_model)
        result = all_reviews["hits"]["hits"]
        scroll = all_reviews["_scroll_id"]
        # handle the rest of the reviews
        while len(result) != 0:
            res = es.scroll(scroll_id=scroll, scroll="1m")
            cleaned_reviews = clean_data(res)
            handle_reviews(cleaned_reviews, data, aspect_model, sentiment_model)
            scroll = res["_scroll_id"]
            result = res["hits"]["hits"]

def handle_reviews(reviews, data, aspect_model, sentiment_model):
    polarity_dict = {1: "positive", 2: "negative", 0: "neutral"}
    for review in reviews:
        # chunk it to sentneces
        sentences = pre_processing(review["review"])
        categories = []
        # for each sentence
        for sentence in sentences:
            # apply the first model (aspect extractor)
            words = extract_aspect(aspect_model, sentence)
            # print(result)
            # if there is not any aspect detected then ignore the sentence
            if not words:
                continue
            # apply the second model (sentiment analyzer)
            data.parse_data([sentence])
            polarities = sentiment_model.predict(data)
            # print(classification)
            for word in words:
                # if the word is detected as aspect
                if words[word] == "B-A":
                    category = None
                    aspect_tag = None
                    for aspect in ASPECTS:
                        if word in ASPECTS[aspect]:
                            if aspect in categories:
                                continue
                            category = aspect
                            aspect_tag = word
                            if word not in polarities:
                                print(
                                    "sentiment model did not give prediction for this aspect"
                                )
                                break
                            categories.append(aspect)
                            sentence_to_index = dict()
                            sentence_to_index["asin"] = review["asin"]
                            sentence_to_index["reveiw_id"] = review["id"]
                            sentence_to_index["text"] = sentence
                            sentence_to_index["aspect"] = category
                            sentence_to_index["aspect_term"] = aspect_tag
                            sentence_to_index["polarity"] = polarity_dict[
                                polarities[word]
                            ]

                            es.index(
                                index="sentences",
                                doc_type="sentence",
                                body=sentence_to_index,
                            )
                            break


if __name__ == "__main__":
    main()
