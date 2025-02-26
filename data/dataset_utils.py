import numpy as np
import torch
import random
import os
from datasets import Dataset, load_dataset, Features, Value, Sequence
import re

seed_setting = 1000
random.seed(seed_setting)
np.random.seed(seed_setting)

home_directory = os.getcwd()
dataset_remove_columns = {
    'c4': ["text"],
    'math': ["problem", "solution"],
    'alpaca': ["instruction", "input", "output"],
    'sharegpt': ["text"], 
    'sharegpt_human_gpt': ["human", "gpt"], 
    'xsum': ["document", "summary"],
    'squad': ["id", "title", "context", "question", "answers"],
    'piqa': ["goal", "sol1", "sol2", "label"],
    'gsm8k': ["question", "answer"],
    'triviaqa': ["question", "answer"],
}

# Define the features schema
ft = Features({
    "id": Value("int64"),
    "context": Value("string"),
    "input": Value("string"),
    "answer": Sequence(Value("string")),
    "options": Sequence(Value("string"))
})

DATASETS = {
    'c4': lambda: load_dataset('json', data_files=home_directory + '/../data/c4-train.00000-of-01024.json', split='train'),
    'math': lambda: load_dataset('json', data_files=home_directory + '/../data/math_pretrain_problem_style.json', split='train'),
    'alpaca': lambda: load_dataset('json', data_files=home_directory + '/../data/alpaca_data_cleaned.json', split='train'),
    'sharegpt': lambda: load_dataset('json', data_files=home_directory + '/../data/ShareGPT_V3_filtered_shuffled.json', split='train'), 
    'sharegpt_human_gpt': lambda: load_dataset('json', data_files=home_directory + '/../data/ShareGPT_V3_filtered_shuffled_human_gpt.json', split='train'), 
    # 'c4': lambda: load_dataset('c4', 'en', split='train'),
    'xsum': lambda: load_dataset('xsum', 'main', split='validation'),
    'ruler': lambda: load_dataset('MaxJeblick/Ruler', split='validation'),
    'infinitebench': lambda: load_dataset("xinrongzhang2022/InfiniteBench", features=ft),
    'squad': lambda: load_dataset('squad', split='validation'),
    'piqa': lambda: load_dataset('piqa', split='validation'),
    'gsm8k': lambda: load_dataset('gsm8k', 'main', split='train'),
    'triviaqa': lambda: load_dataset('trivia_qa', 'rc.nocontext', split='train'),
}

def digit_or_text(input):
    # Define a dictionary mapping digits and text representations to their corresponding values
    digit_dict = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six',
        '7': 'seven', '8': 'eight', '9': 'nine', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
        'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'three hundred sixty': '360',
        # common fractions
        '1/4': 'one quarter', '1/2': 'one half', '1/3': 'one third', '2/3': 'two thirds', '3/4': 'three quarters',
        '4/5': 'four fifths', 'one quarter': '1/4', 'half': '50%', 'one half': '1/2', 'one third': '1/3',
        'two thirds': '2/3', 'three quarters': '3/4', 'four fifths': '4/5'
    }
    
    # Check if the input is a single digit number (either as a string)
    if input.lower() in digit_dict:
        return digit_dict[input.lower()]
    # If the input is not a single digit number, return None
    else:
        return None
    

def load_data(logger, dataset_name, tokenizer, max_new_tokens, n_samples=128):
    raw_data = DATASETS[dataset_name]()
    # logger.info(raw_data.features)
    # input()
    raw_data = raw_data.shuffle(seed=seed_setting).select(range(min(n_samples, len(raw_data))))
    logger.info(f'tokenizer.model_max_length: {tokenizer.model_max_length}')

    def dummy_gen():
        return raw_data

    def tokenize_alpaca(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        input_token_length = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                text = prompt + opt
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
                text = prompt + opt
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length - max_new_tokens:
                continue

            tokenized_data = tokenizer(text)
            # added by yujie
            input_token_length.append(len(tokenized_data["input_ids"]))

            input_ids.append(
                tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(
                tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        logger.info(f'input_ids_length: {input_token_length}')

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts
        }

    def tokenize_math_c4_sharegpt(examples):
        texts = examples["text"]
        prompts = []
        input_ids = []
        attention_mask = []
        input_token_length = []
        for text in texts:

            if len(tokenizer(text)["input_ids"]) >= tokenizer.model_max_length - max_new_tokens:
                continue

            tokenized_data = tokenizer(text)
            # added by yujie
            input_token_length.append(len(tokenized_data["input_ids"]))

            input_ids.append(
                tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(
                tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(text)

        logger.info(f'input_ids_length: {input_token_length}')

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts
        }

    def tokenize_squad(examples):
        contexts = examples["context"]
        questions = examples["question"]
        answers = examples["answers"]
        ids = examples["id"]

        input_ids = []
        attention_mask = []
        ref_answers = []
        sample_ids = []
        input_token_length = []
        for cont, quest, ans, sample_id in zip(contexts, questions, answers, ids):
            
            # modify ground truth to accept text and numeric single-digits
            ans_text = ans['text']
            ans_start = ans['answer_start']
            for i in range(len(ans['text'])):
                val = digit_or_text(ans['text'][i])
                if val:
                    ans_text.append(val)
                    ans_start.append(ans['answer_start'][i])
            
            ref_ans = {'answers': {'answer_start': ans_start, 'text': ans_text}, 'id': sample_id}                    
            prompt = f"Answer the question based on the context below.\nContext: {cont}\nQuestion: {quest}\nAnswer:"
            
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length - max_new_tokens:
                continue

            tokenized_data = tokenizer(prompt)
            # added by yujie
            input_token_length.append(len(tokenized_data["input_ids"]))

            input_ids.append(
                tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(
                tokenized_data["attention_mask"][: tokenizer.model_max_length])
            ref_answers.append(ref_ans)
            sample_ids.append(sample_id)

        logger.info(f'input_ids_length: {input_token_length}')

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ref_answers": ref_answers,
            "sample_ids": sample_ids,
        }
    
    def tokenize_piqa(examples):
        goals = examples["goal"]
        sol1s = examples["sol1"]
        sol2s = examples["sol2"]
        label_list = examples["label"]

        input_ids = []
        attention_mask = []
        labels = []
        input_token_length = []
        for goal, sol1, sol2, label in zip(goals, sol1s, sol2s, label_list):
            # prompt = f"Which of the following solutions is more appropriate for achieving the goal?\n\nGoal: {goal}\n\nSolution 1: {sol1}\n\nSolution 2: {sol2}\n\nAnswer (1 or 2):"
            prompt = f"Goal is: {goal}\nSolution 1: {sol1}\nSolution 2: {sol2}\nCorrect: Solution "

            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length - max_new_tokens:
                continue

            tokenized_data = tokenizer(prompt)
            # added by yujie
            input_token_length.append(len(tokenized_data["input_ids"]))

            input_ids.append(
                tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(
                tokenized_data["attention_mask"][: tokenizer.model_max_length])
            labels.append(label)

        logger.info(f'input_ids_length: {input_token_length}')

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels,
        }

    dataset = Dataset.from_generator(dummy_gen)
    
    dataset_tokenize = {
        'c4': tokenize_math_c4_sharegpt,
        'math': tokenize_math_c4_sharegpt,
        'alpaca': tokenize_alpaca,
        'sharegpt': tokenize_math_c4_sharegpt, 
        'squad': tokenize_squad,
        'piqa': tokenize_piqa,
    }

    dataset = dataset.map(
        dataset_tokenize[dataset_name],
        batched=True,
        batch_size=len(dataset),
        num_proc=5,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=dataset_remove_columns[dataset_name]
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])
    
    return dataset


def load_data_text(dataset_name, n_samples=1024):
    raw_data = DATASETS[dataset_name]()
    # print(raw_data.features, len(raw_data))
    # input()
    raw_data = raw_data.shuffle(seed=seed_setting).select(range(min(n_samples, len(raw_data))))
    
    def dummy_gen():
        return raw_data

    def extract_alpaca(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        texts = []
        ref_answers = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                text = prompt + opt
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
                text = prompt + opt
            # texts.append(text)
            ref_answers.append(opt)
            texts.append(prompt)

        return {
            "prompt": texts,
            "ref_answers": ref_answers,
        }

    def extract_math(examples):
        problems = examples["problem"]
        solutions = examples["solution"]

        texts = []
        ref_answers = []
        for probl, sol in zip(problems, solutions):
            prompt = f"Problem:\n{probl}\nSolution:\n"
            # texts.append(text)
            ref_answers.append(sol)
            texts.append(prompt)

        return {
            "prompt": texts,
            "ref_answers": ref_answers,
        }
    
    def extract_sharegpt_human_gpt(examples):
        humans = examples["human"]
        gpts = examples["gpt"]

        texts = []
        ref_answers = []
        for human, gpt in zip(humans, gpts):
            texts.append(human)
            ref_answers.append(gpt)

        return {
            "prompt": texts,
            "ref_answers": ref_answers,
        }

    def extract_xsum(examples):
        problems = examples["document"]
        solutions = examples["summary"]

        texts = []
        ref_answers = []
        for probl, sol in zip(problems, solutions):
            prompt = f"Document:\n{probl}\nSummary:\n"
            # texts.append(text)
            ref_answers.append(sol)
            texts.append(prompt)

        return {
            "prompt": texts,
            "ref_answers": ref_answers,
        }
    
    def extract_c4_sharegpt(examples):
        texts = examples["text"]
        ref_answers = texts

        return {
            "prompt": texts,
            "ref_answers": ref_answers,
        }

    def extract_squad(examples):
        contexts = examples["context"]
        questions = examples["question"]
        answers = examples["answers"]
        ids = examples["id"]

        texts = []
        ref_answers = []
        sample_ids = []
        for cont, quest, ans, sample_id in zip(contexts, questions, answers, ids):
            # modify ground truth to accept text and numeric single-digits
            ans_text = ans['text']
            ans_start = ans['answer_start']
            for i in range(len(ans['text'])):
                val = digit_or_text(ans['text'][i])
                if val:
                    ans_text.append(val)
                    ans_start.append(ans['answer_start'][i])
            
            ref_ans = {'answers': {'answer_start': ans_start, 'text': ans_text}, 'id': sample_id}                    
            prompt = f"Answer the question based on the context below.\nContext: {cont}\nQuestion: {quest}\nAnswer:"

            texts.append(prompt)
            ref_answers.append(ref_ans)
            sample_ids.append(sample_id)

        return {
            "prompt": texts,
            "ref_answers": ref_answers,
            "sample_ids": sample_ids,
        }
    
    def extract_gsm8k(examples):
        questions = examples["question"]
        answers = examples["answer"]

        texts = []
        ref_answers = []
        for quest, ans in zip(questions, answers):
            prompt = f"Question: {quest}\nAnswer:"
            texts.append(prompt)
            ref_answers.append(ans)

        return {
            "prompt": texts,
            "ref_answers": ref_answers,
        }
    
    def extract_piqa(examples):
        goals = examples["goal"]
        sol1s = examples["sol1"]
        sol2s = examples["sol2"]
        label_list = examples["label"]

        texts = []
        ref_answers = []
        for goal, sol1, sol2, label in zip(goals, sol1s, sol2s, label_list):
            # prompt = f"Which of the following solutions is more appropriate for achieving the goal?\n\nGoal: {goal}\n\nSolution 1: {sol1}\n\nSolution 2: {sol2}\n\nAnswer (1 or 2):"
            prompt = f"Goal is: {goal}\nSolution 1: {sol1}\nSolution 2: {sol2}\nCorrect: Solution "
            texts.append(prompt)
            ref_answers.append(label)

        return {
            "prompt": texts,
            "ref_answers": ref_answers,
        }

    dataset = Dataset.from_generator(dummy_gen)
    
    dataset_extract = {
        'c4': extract_c4_sharegpt,
        'math': extract_math,
        'alpaca': extract_alpaca,
        'sharegpt': extract_c4_sharegpt, 
        'sharegpt_human_gpt': extract_sharegpt_human_gpt,
        'xsum': extract_xsum,
        'squad': extract_squad,
        'piqa': extract_piqa,
        'gsm8k': extract_gsm8k,
        'triviaqa': extract_gsm8k,
    }

    dataset = dataset.map(
        dataset_extract[dataset_name],
        batched=True,
        batch_size=len(dataset),
        num_proc=5,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=dataset_remove_columns[dataset_name]
    )

    dataset = dataset.to_list()

    return dataset


def piqa_prediction(prediction_text):
    # prediction_text = prediction_text.lower()
    ptr = re.compile(r'\A1.*\n')
    ptr = ptr.findall(prediction_text)
    # print(ptr)

    if len(ptr) > 0:
        label = 0
    else:
        label = 1

    return label
