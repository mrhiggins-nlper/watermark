import sys
sys.path.append('/watermark/lm-watermarking')

import os
import torch  
import json  
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList  
from extended_watermark_processor import WatermarkLogitsProcessor   
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print(f"Using device: {device}")  
    
MODEL_DIR = "01-ai/Yi-6B"  
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto").to(device)  
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)  
    
watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()), gamma=0.25, delta=2.0, seeding_scheme="selfhash")  
    
def generate_articles(text):  
    tokenized_input = tokenizer(text, return_tensors="pt").to(device)  
    output = model.generate(**tokenized_input, max_new_tokens = 100)  
    output = output[:, tokenized_input["input_ids"].shape[-1]:]  
    output = tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)[0]   
    return output  
    
def generate_articles_watermarked(text):  
    tokenized_input = tokenizer(text, return_tensors="pt").to(device)  
    output = model.generate(**tokenized_input, max_new_tokens = 100, logits_processor=LogitsProcessorList([watermark_processor]))  
    output = output[:, tokenized_input["input_ids"].shape[-1]:]  
    output = tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)[0]    
    return output  
    
with open('watermark/key_words/key_words.json', 'r', encoding='utf-8') as file:  
    keywords = json.load(file)  
  
articles = []  
  
for area, area_keywords in keywords.items():  
    article = {area: []}  
    for keyword in area_keywords[:10]:  
        keyword_article = {  
            "keyword": keyword,  
            "without_watermark":None, 
            "with_watermark":None 
        }  
          
        prompt = f"Please give me the definition of the word: {keyword} in the area: {area}. {keyword} is:"    
            
        definition = generate_articles(prompt)  
            
        definition_watermarked = generate_articles_watermarked(prompt)   
        keyword_article["with_watermark"] = definition_watermarked  
        keyword_article["without_watermark"] = definition  
        article[area].append(keyword_article) 
        
    articles.append(article)  
    print(f"Area {area} is completed.") 
  
    with open(f"article_{area}.json", "w") as file:  
        json.dump(article, file, ensure_ascii=False, indent=4)

articles_dir = '/watermark/Yi-6B_data/articles'  
   
all_articles = []  
   
for filename in os.listdir(articles_dir):  
    if filename.endswith('.json'):    
        file_path = os.path.join(articles_dir, filename)  
            
        with open(file_path, 'r', encoding='utf-8') as file:  
            articles = json.load(file)  
            
            for area, area_articles in articles.items():  
                all_articles.extend(area_articles)  
    
output_file_path = '/watermark/Yi-6B_data/articles/articles.json' 
with open(output_file_path, 'w', encoding='utf-8') as output_file:  
    json.dump(all_articles, output_file, ensure_ascii=False, indent=4)
