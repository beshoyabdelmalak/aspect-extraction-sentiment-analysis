from elasticsearch import Elasticsearch

es = Elasticsearch()


def clean_data(docs):
    outputProducts = []
    for hit in docs["hits"]["hits"]:
        item = dict()
        item["asin"] = hit["_source"]["asin"]
        item["reveiw_id"] = hit["_source"]["reveiw_id"]
        item["text"] = hit["_source"]["text"]
        item["aspect"] = hit["_source"]["aspect"]
        item["aspect_term"] = hit["_source"]["aspect_term"]
        item["polarity"] = hit["_source"]["polarity"]
        outputProducts.append(item)
    return outputProducts


def main():
    all_sentences = es.search(
        index="sentences", body={"size": 1000, "query": {"match_all": {}}}
    )
    cleaned_data = clean_data(all_sentences)
    products = dict()
    for sentence in cleaned_data:
        if sentence["asin"] in products:
            if sentence["aspect"] in products[sentence["asin"]]:
                if sentence["polarity"] == "positive":
                    products[sentence["asin"]][sentence["aspect"]]["positive"] += 1
                elif sentence["polarity"] == "negative":
                    products[sentence["asin"]][sentence["aspect"]]["negative"] += 1
                elif sentence["polarity"] == "neutral":
                    products[sentence["asin"]][sentence["aspect"]]["neutral"] += 1
            else:
                if sentence["polarity"] == "positive":
                    products[sentence["asin"]][sentence["aspect"]] = {
                        "positive": 1,
                        "negative": 0,
                        "neutral": 0,
                    }
                elif sentence["polarity"] == "negative":
                    products[sentence["asin"]][sentence["aspect"]] = {
                        "positive": 1,
                        "negative": 0,
                        "neutral": 0,
                    }
                elif sentence["polarity"] == "neutral":
                    products[sentence["asin"]][sentence["aspect"]] = {
                        "positive": 1,
                        "negative": 0,
                        "neutral": 0,
                    }
        else:
            if sentence["polarity"] == "positive":
                products = {
                    sentence["asin"]: {
                        sentence["aspect"]: {"positive": 1, "negative": 0, "neutral": 0}
                    }
                }
            elif sentence["polarity"] == "negative":
                products = {
                    sentence["asin"]: {
                        sentence["aspect"]: {"positive": 1, "negative": 0, "neutral": 0}
                    }
                }
            elif sentence["polarity"] == "neutral":
                products = {
                    sentence["asin"]: {
                        sentence["aspect"]: {"positive": 1, "negative": 0, "neutral": 0}
                    }
                }

    for product in products:
        products[product]["asin"] = product
        es.index(index="products", doc_type="product", body=products[product])


if __name__ == "__main__":
    main()
