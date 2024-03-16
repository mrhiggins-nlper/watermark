import json  
    
with open('evaluate_articles/evaluated_articles.json', 'r', encoding='utf-8') as file:  
    without_watermark = json.load(file)   

def f1_score(without_watermark, with_watermark):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    if result["with_watermark"]["prediction"] == True:
        TP += 1
    else:
        FN += 1
    
    if result["without_watermark"]["prediction"] == False:
        TN += 1
    else:
        FP += 1
    
    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return acc, precision, recall, f1_score

data = f1_score(without_watermark, with_watermark)
f1_score = {"accuracy": data[0], "precision": data[1], "recall": data[2], "F1_score": data[3]}

with open('evaluation_results.json', 'w', encoding='utf-8') as file:  
    json.dump(f1_score, file, ensure_ascii=False, indent=4)

    
            
