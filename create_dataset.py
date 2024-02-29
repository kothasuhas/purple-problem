import argparse
from tqdm import tqdm
import json
import re
from openai import OpenAI
import numpy as np

def get_prompts():
    responses = []
    topics = ["scientific", "fairytale", "fantasy", "cultural", "historical", "food", "scary", "economic", "sports", "celebrity", "nonsense", "romantic", "funny", "philosophical", "space", "nature", "children"]
    for i in tqdm(range(args.dataset_length)):
        topic = topics[i % len(topics)]
        messages = [{"role": "user", "content": f"Create 50 {topic} questions with the word '{args.word}'"}]
        
        chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
        temperature=1.0,
        n=1,
        frequency_penalty=1.0,
        )
        
        response = chat_completion.choices[0].message.content
        response = response.split('\n')
        response = [re.sub(r'[0-9"]', '', text) for text in response]
        response = [text.lstrip('. ') for text in response]
        response = [text for text in response if text != ""]

        for text in response:
            try:
                if isinstance(text, str) and text.lower() == 'nan':
                    continue
                elif np.isnan(text):
                    continue
            except TypeError:
                pass
            if '?' not in text:
                continue
            if any(word in text.lower() for word in ['sorry', 'note', 'apologize', 'error:']):
                continue
            elif ':' in text:
                continue

            if text.startswith(')'):
                text = text.lstrip(')')

            responses.append(text)

    return responses

def create_responses(prompts):
    responses = []
    for prompt in tqdm(prompts):
        messages = [{"role": "user", "content": f"{prompt} Give one answer containing '{args.word}' and one answer without '{args.word}', strictly. Mark them as 'Answer A' and 'Answer B', respectively. Each answer must be strictly less than 30 words."}]
        
        chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
        temperature=1.0,
        n=1,
        frequency_penalty=1.0,
        )

        response = chat_completion.choices[0].message.content
        response = response.split('\n')
        response = [text.split(': ', 1)[-1].strip() for text in response]
        responses.append(response)

    return responses

def create_dataset(prompts, responses, split_function):
    dataset_dict = dict(
        prompt=[],
        chosen=[],
        rejected=[]
    )

    for prompt, response in zip(prompts, responses):
        safe_response, harmful_response = split_function(response, args.word)

        if safe_response and harmful_response and safe_response != "" and harmful_response != "":
            dataset_dict['prompt'].append(prompt)
            dataset_dict['chosen'].append(safe_response)
            dataset_dict['rejected'].append(harmful_response)

    print(len(dataset_dict['prompt']))
    dataset_name = args.word

    length = len(dataset_dict['chosen'])
    indices = list(range(length))
    np.random.shuffle(indices)
    
    indices1 = indices[:int(length*0.8)]
    indices2 = indices[int(length*0.8):int(length*0.8 + length*0.12)]
    indices3 = indices[int(length*0.8 + length*0.12):]

    dataset_dict1 = {key: [value[i] for i in indices1] for key, value in dataset_dict.items()}
    dataset_dict2 = {key: [value[i] for i in indices2] for key, value in dataset_dict.items()}
    dataset_dict3 = {key: [value[i] for i in indices3] for key, value in dataset_dict.items()}

    with open(f'datasets/{dataset_name}_dataset_train.json', 'w') as f:
        json.dump(dataset_dict1, f)
    with open(f'datasets/{dataset_name}_dataset_validation.json', 'w') as f:
        json.dump(dataset_dict2, f)
    with open(f'datasets/{dataset_name}_dataset_test.json', 'w') as f:
        json.dump(dataset_dict3, f)

def split_on_word(responses, word):
    safe_response = None
    harmful_response = None
    for response in responses:
        if word in response.lower():
            harmful_response = response
        else:
            safe_response = response
        if safe_response and harmful_response:
            break

    if safe_response is None or safe_response == "" or harmful_response is None or harmful_response == "":
        return None, None
    else:
        return safe_response, harmful_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--dataset_length", type=int, default=200) # multiplied by 50
    parser.add_argument("-w", "--word", type=str, nargs='+', default='purple')
    args = parser.parse_args()

    client = OpenAI(
    api_key='API-KEY' #Enter own OpenAI API KEY
    )

    prompts = get_prompts()
    responses = create_responses(prompts)
    create_dataset(prompts, responses, split_on_word)
