from zhipuai import ZhipuAI  
import re  
import json

with open('areas.json', 'r', encoding='utf-8') as file:  
    areas = json.load(file)  

def generate(text):  
    client = ZhipuAI(api_key="your_api_key_here")  
    response = client.chat.completions.create(  
        model="glm-4",  
        messages=[  
            {"role": "user", "content": text},  
        ],  
    )  
    response = response.choices[0].message  
    return response.content  


def generate_keywords(word):  
    keywords_list = []  
    while len(keywords_list) < 50:  
        prompt_generate_keywords = "I need 70 keywords on the scientific research on " + word + " separated by only commas. Do not generate anything else."  
        keywords = generate(prompt_generate_keywords)  
        new_keywords_list = re.split(r'[，,]', keywords)  
          
        cleaned_new_keywords_list = [keyword.strip().replace('.', '').lower() for keyword in new_keywords_list if not re.search(r'[\u4e00-\u9fff]', keyword)]  
        
        keywords_list += cleaned_new_keywords_list[5:-5]  
        keywords_list = list(dict.fromkeys(keywords_list))    
        if len(keywords_list) >= 50:  
            break  
  
    keywords_list = keywords_list[:50] 
    return keywords_list 

keywords_dict = {}  

for area in areas:  
    keywords = generate_keywords(area)  
    keywords_dict[area] = keywords  
    print(f"{areas.index(area) + 1} / {len(areas)} is completed.")
    
with open("key_words.json", "w", encoding="utf-8") as file:  
    json.dump(keywords_dict, file, ensure_ascii=False, indent=4)



