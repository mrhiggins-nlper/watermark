import sys
sys.path.append('watermark/lm-watermarking')

from extended_watermark_processor import WatermarkDetector
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

with open('/watermark/Yi-6B_data/articles/articles.json', 'r', encoding='utf-8') as file:
    articles = json.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_DIR = "01-ai/Yi-6B"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto").to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                    gamma=0.25,
                    seeding_scheme="selfhash",
                    device=model.device,
                    tokenizer=tokenizer,
                    z_threshold=4.0,
                    normalizers=[],
                    ignore_repeated_ngrams=True)

score_dicts = []

def convert_to_list(data):
    if isinstance(data, dict):
        return {k: convert_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_list(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.tolist()
    else:
        return data

for article in articles:
    try:
        score_without_watermark = convert_to_list(watermark_detector.detect(article["without_watermark"]))
        score_with_watermark = convert_to_list(watermark_detector.detect(article["with_watermark"]))
    except ValueError:
        continue
    score = {"keyword":article["keyword"], "article_without_watermark":article["without_watermark"], "score_without_watermark":score_without_watermark, 
             "article_with_watermark":article["with_watermark"], "score_without_watermark":score_with_watermark}
    score_dicts.append(score)

with open('evaluated_articles.json', 'w', encoding='utf-8') as file:
    json.dump(score_dicts, file, ensure_ascii=False, indent=4)




        
        

    
