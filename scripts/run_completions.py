# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import pathlib
import os

from npueval.aiecoder import AIECoder

#os.environ["OPENAI_API_KEY"] = "sk-xxx"

with open("dataset/npueval.jsonl", 'r') as f:
    tests = [json.loads(line) for line in f]

def generate_one_completion(prompt, model="gpt-4.1", base_url=None, temperature=0.0, top_p=1.0, attempts=1):
    coder = AIECoder(model=model, temperature=temperature, top_p=top_p, base_url=base_url, attempts=attempts)
    response = coder(prompt)
    
    result = {
        "code": coder.extract_codeblock(response['response']),
        "stats": {"token_usage": response['token_usage'],
                  "history": response['history']}
    }

    return result


# Proprietary models
N = [1, 2] # 1 - no compile just first pass, 2+ - retry with compiler
models = ["gpt-4o-mini", "gpt-4.1"]
for MODEL in models:
    for attempts in N:
        print(f"{MODEL} N={attempts}")
        solutions_path = f"results/solutions/{MODEL}_attempts_{attempts}"
        pathlib.Path(solutions_path).mkdir(parents=True, exist_ok=True)
        for idx, test in enumerate(tests):
            if os.path.isfile(f"{solutions_path}/{test['kernel_name']}.json"):
                print(f"{solutions_path}/{test['kernel_name']}.json already exists, skipping...")
                continue
            print(f"Generating solution for {test['kernel_name']}")
            response = generate_one_completion(test['prompt'], model=MODEL, attempts=attempts, temperature=0, top_p=1.0)
            with open(f"{solutions_path}/{test['kernel_name']}.json", 'w') as file:
                json.dump(response, file, indent=4)

# With RAG (basic llama-index retriever)
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    SimpleDirectoryReader
)
    
PERSIST_DIR = "./rag/vector_database"

num_retrieved = [1] # change this to increase number of retrieved kernel examples
for MODEL in models:
    for attempts in N:
        for k in num_retrieved:
            print(f"{MODEL} N={attempts} k={k}")
            
            if not os.path.exists(PERSIST_DIR):
                print("Indexing...")
                documents = SimpleDirectoryReader("rag/kernels", recursive=True).load_data()
                index = VectorStoreIndex.from_documents(documents)
                index.storage_context.persist(persist_dir=PERSIST_DIR)
            else:
                storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                index = load_index_from_storage(storage_context)
            
            retriever = index.as_retriever(similarity_top_k=k)
            
            solutions_path = f"results/solutions/{MODEL}_attempts_{attempts}_rag_{k}"
            pathlib.Path(solutions_path).mkdir(parents=True, exist_ok=True)
            for idx, test in enumerate(tests):
                if os.path.isfile(f"{solutions_path}/{test['kernel_name']}.json"):
                    print(f"{solutions_path}/{test['kernel_name']}.json already exists, skipping...")
                    continue
                print(f"Generating solution for {test['kernel_name']}")
                nodes = retriever.retrieve(test['prompt'])
                context_string = "Reference vectorized code:\n"
                for node in nodes:
                    context_string += node.node.text
            
                prompt_with_context = test['prompt'] + "\n" + context_string
                #print(prompt_with_context)
                response = generate_one_completion(prompt_with_context, model=MODEL, attempts=attempts, temperature=0, top_p=1.0)
                with open(f"{solutions_path}/{test['kernel_name']}.json", 'w') as file:
                    json.dump(response, file, indent=4)
