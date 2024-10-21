import json
import os
import pickle
import random
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi

seed = 2024
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrievalBM25:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        """
        Initialize with BM25
        """
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # unique contexts
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Tokenize all contexts using the provided tokenizer function
        self.tokenized_contexts = [tokenize_fn(text) for text in self.contexts]

        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_contexts)

    def get_sparse_embedding(self):
        """
        No need for embedding storage with BM25.
        """
        pass

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Retrieve top-k passages based on BM25 scores
        """
        if isinstance(query_or_dataset, str):
            tokenized_query = self.bm25.tokenizer(query_or_dataset)
            doc_scores = self.bm25.get_scores(tokenized_query)
            topk_indices = np.argsort(doc_scores)[::-1][:topk]

            print("[Search query]\n", query_or_dataset, "\n")
            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[topk_indices[i]]:4f}")
                print(self.contexts[topk_indices[i]])

            return doc_scores[topk_indices], [self.contexts[i] for i in topk_indices]

        elif isinstance(query_or_dataset, Dataset):
            total = []
            for example in tqdm(query_or_dataset, desc="BM25 retrieval:"):
                tokenized_query = self.bm25.tokenizer(example["question"])
                doc_scores = self.bm25.get_scores(tokenized_query)
                topk_indices = np.argsort(doc_scores)[::-1][:topk]
                
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([self.contexts[i] for i in topk_indices]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)