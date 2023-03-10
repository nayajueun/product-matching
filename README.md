# product-matching
# product-matching

The project aims to carry out the data preprocessing, training and inference of a model which can match two products in different catalogue based on the text data.

# Example:

Product 1: <br />
`{"title_left":"495906 b21 hp x5560 2 80ghz ml350 g6 , null new wholesale price"`<br />
`"description_left":"description intel xeon x5560 ml350 g6 2 80ghz 4 core 8mb 95w full processor option kitpart number s option part 495906 b21",` <br />
`"brand_left":"hp enterprise", `<br />
`"specTableContent_left":"specifications category proliant processor sub category xeon generation g6 part number 495906 b21 products id 459573 product type processor upgrade processor type intel xeon processor core quad core processor qty 1 clock speed 2 8ghz bus speed 1333mhz l2 cache 1mb l3 cache 8mb 64 bit processing yes process technology 45nm processor socket socket b lga 1366 thermal design power 95w",`<br />
`"keyValuePairs_left":{"category":"proliant processor","sub category":"xeon","generation":"g6","part number":"495906 b21","products id":"459573","product type":"processor upgrade","processor type":"intel xeon","processor core":"quad core","processor qty":"1","clock speed":"2 8ghz","bus speed":"1333mhz","l2 cache":"1mb","l3 cache":"8mb","64 bit processing":"yes","process technology":"45nm","processor socket":"socket b lga 1366","thermal design power":"95w"},`<br />
`"category_left":"Computers_and_Accessories",`<br />
<br />

Product 2:<br />
`"title_right":"495906 b21 hp x5560 2 80ghz ml350 g6",`<br />
`"description_right":"description intel xeon x5560 ml350 g6 2 80ghz 4 core 8mb 95w full processor option kitpart number s part 495906 b21",`<br />
`"brand_right":"hp enterprise",`<br />
`"specTableContent_right":"specifications category proliant processor sub category xeon generation g6 part number 495906 b21 products id 459573 product type processor upgrade processor type intel xeon processor core quad core processor qty 1 clock speed 2 8ghz bus speed 1333mhz l2 cache 1mb l3 cache 8mb 64 bit processing yes process technology 45nm processor socket socket b lga 1366 thermal design power 95w",`<br />
`"keyValuePairs_right":{"category":"proliant processor","sub category":"xeon","generation":"g6","part number":"495906 b21","products id":"459573","product type":"processor upgrade","processor type":"intel xeon","processor core":"quad core","processor qty":"1","clock speed":"2 8ghz","bus speed":"1333mhz","l2 cache":"1mb","l3 cache":"8mb","64 bit processing":"yes","process technology":"45nm","processor socket":"socket b lga 1366","thermal design power":"95w"},`
`"category_right":"Computers_and_Accessories"`<br />

If these two products are different, output = 0.<br />
If they are the same, output = 1
(In the case above, output = 1)

* Sample data source: https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/trainsets/


## Implemented in pipeline.py:
- Finetuning SBERT with description data
- Fit text features relevant to semantic/lexical similarity with TfIdf vectorizer, and those relevant to paraphrase identification with SBERT encoder.
- After computing numerical features using cosine similarity, train the data with XGBoost classifier

## Dependency:
- Pre-trained SBERT model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
