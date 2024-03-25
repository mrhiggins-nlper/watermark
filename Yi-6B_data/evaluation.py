import json  
import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

with open('evaluate_articles/evaluated_articles.json', 'r', encoding='utf-8') as file:  
    evaluated_articles = json.load(file)   

with open('key_words.json', 'r', encoding='utf-8') as file:  
    key_words = json.load(file)  

with open('evaluation_results_overlap.json', 'r', encoding='utf-8') as file:  
    evaluation_results_overlap = json.load(file)

def f1_score(articles):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for result in articles:
        if result["score_with_watermark"]["prediction"] == True:
            TP += 1
        else:
            FN += 1
    
        if result["score_without_watermark"]["prediction"] == False:
            TN += 1
        else:
            FP += 1
    
    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return acc, precision, recall, f1_score

data = f1_score(evaluated_articles)
f1_score = {"accuracy": data[0], "precision": data[1], "recall": data[2], "F1_score": data[3]}

def calculate_overlap_rate(text_1, text_2):    
    words_1 = re.findall(r'\b\w+\b', text_1.lower())  
    words_2 = re.findall(r'\b\w+\b', text_2.lower())  
        
    counter_1 = Counter(words_1)  
    counter_2 = Counter(words_2)  
       
    common_words = counter_1 & counter_2  
        
    overlap_count = sum(common_words.values())  
    text_1_count = len(words_1)  
       
    if text_1_count == 0:  
        return 0  
      
    overlap_rate = overlap_count / text_1_count  
    return overlap_rate

overlaps = []
without_watermark = ""
with_watermark = ""

for area, key_words_in_area in key_words.items():
    area_overlap = {area:{}}
    for result in evaluated_articles:
        if result["keyword"] in key_words_in_area:
            overlap_rate = calculate_overlap_rate(result["article_without_watermark"], result["article_with_watermark"])
            area_overlap[area][result["keyword"]] = overlap_rate
            without_watermark = without_watermark + result["article_without_watermark"]
            with_watermark = with_watermark + result["article_with_watermark"]
    overlaps.append(area_overlap)

overall_overlap = calculate_overlap_rate(without_watermark, with_watermark)

predictions = []
for area, key_words_in_area in key_words.items():
    area_prediction_result = {area:{}}
    without_watermark = ""
    with_watermark = ""
    for result in evaluated_articles:
        if result["keyword"] in key_words_in_area:
            area_prediction_result[area][result["keyword"]] = ((result["score_with_watermark"]["prediction"], len(word_tokenize(result["article_with_watermark"]))), 
                                                               (result["score_without_watermark"]["prediction"], len(word_tokenize(result["article_without_watermark"]))))
    predictions.append(area_prediction_result)

with open('evaluation_results_f1_score.json', 'w', encoding = 'utf-8') as file:  
    json.dump(f1_score, file, ensure_ascii=False, indent=4)

with open('evaluation_results_predictions.json', 'w', encoding = 'utf-8') as file:  
    json.dump(predictions, file, ensure_ascii=False, indent=4)

categories = []
overlaps = []
sub_categories = []

for result in evaluation_results_overlap:
    category = list(result.keys())[0]
    categories.append(category)
    overlaps.append(result['overall_overlap'])
    sub_categories.extend([(category, sub_cat, result[category][sub_cat]) for sub_cat in result[category]])

average_overlap = np.mean(overlaps)

plt.figure(figsize=(10, 6))

for i, (category, sub_category, overlap) in enumerate(sub_categories):
    color = 'red' if sub_category == category else 'grey'
    plt.scatter(categories.index(category), overlap, color=color)

plt.scatter(range(len(categories)), overlaps, color='red', marker='o')

plt.axhline(y = average_overlap, color='blue', linestyle='--', label='Average Overlap')
plt.axhline(y = overall_overlap, color='green', linestyle='--', label='Overall Overlap')

plt.xticks(range(len(categories)), "")
plt.xlabel('Category')
plt.ylabel('Overlap Rate')
plt.title('Overlap Rate')
plt.grid(True)

plt.tight_layout()
plt.show()







    
            
