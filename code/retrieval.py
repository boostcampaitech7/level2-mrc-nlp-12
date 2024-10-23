import argparse
import json
import os
import pickle
import random
import re
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_from_disk
from kiwipiepy import Kiwi

# 추가한 코드
from rank_bm25 import BM25Okapi
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

seed = 2024
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        index_file: Optional[str] = "bm25_index.pkl",
    ) -> NoReturn:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """
        self.tokenize_fn = tokenize_fn
        self.data_path = data_path
        self.index_file = index_file

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        # 저장된 인덱스 파일이 있으면 로드, 없으면 새로 생성
        if os.path.isfile(self.index_file):
            print("Loading BM25 index from pickle file...")
            with open(self.index_file, "rb") as f:
                self.BM25 = pickle.load(f)
        else:
            print(f"Tokenizing {len(self.contexts)} contexts for BM25...")
            self.tokenized_contexts = [
                tokenize_fn(context) for context in tqdm(self.contexts)
            ]
            self.BM25 = BM25Okapi(self.tokenized_contexts, k1=0.5, b=0.75)
            with open(self.index_file, "wb") as f:
                pickle.dump(self.BM25, f)
            print("BM25 index saved to pickle.")

    def build_faiss(self, num_clusters=64) -> NoReturn:
        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 25
    ) -> Union[Tuple[List[float], List[int]], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset: Query를 받습니다. (문자열 또는 Dataset)
            topk: 상위 몇 개의 passage를 사용할지 지정합니다.

        Returns:
            단일 Query의 경우 상위 passage들을 반환합니다.
            다수의 Query의 경우 DataFrame으로 반환합니다.
        """

        if isinstance(query_or_dataset, str):
            # 단일 쿼리 처리
            tokenized_query = self.tokenize_fn(query_or_dataset)
            doc_scores = self.BM25.get_scores(tokenized_query)
            topk_indices = np.argsort(doc_scores)[::-1][:topk]
            print("[Search query]\n", query_or_dataset, "\n")

            # for i in range(topk):
            #     print(f"Top-{i+1} passage with score {doc_scores[topk_indices[i]]:.4f}")
            #     print(self.contexts[topk_indices[i]])

            return doc_scores[topk_indices].tolist(), topk_indices.tolist()

        elif isinstance(query_or_dataset, Dataset):
            # 여러 쿼리 처리
            total = []

            # 이거 나중에 에러 없는 경우에는 queries[:50] -> queries로 변경하기
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="BM25 retrieval")
            ):
                query = example["question"]
                tokenized_query = self.tokenize_fn(query)
                doc_scores = self.BM25.get_scores(tokenized_query)
                topk_indices = np.argsort(doc_scores)[::-1][:topk]

                tmp = {
                    "question": query,
                    "id": query_or_dataset["id"][idx],
                    "context": " ".join([self.contexts[i] for i in topk_indices]),
                }

                if "answers" in example.keys():
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            # for i in range(topk):
            #     print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
            #     print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()


# Poly Encoder를 이용해서 DenseRetrieval 클래스를 구현
class PolyEncoder(nn.Module):
    def __init__(self, model_name_or_path, poly_m=64, device="cuda"):
        super(PolyEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.poly_m = poly_m
        self.poly_code_embeddings = nn.Embedding(
            poly_m, self.bert.config.hidden_size
        ).to(device)
        self.device = device
        # Extract 'text' field from wiki and remove duplicates

        # Initialize poly_code_embeddings
        nn.init.normal_(
            self.poly_code_embeddings.weight, self.bert.config.hidden_size**-0.5
        )

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        context = self.contexts[idx]

        # 쿼리와 문맥을 토크나이징
        query_inputs = self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        context_inputs = self.tokenizer(
            context,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "query_input_ids": query_inputs["input_ids"].squeeze(0),
            "query_attention_mask": query_inputs["attention_mask"].squeeze(0),
            "context_input_ids": context_inputs["input_ids"].squeeze(0),
            "context_attention_mask": context_inputs["attention_mask"].squeeze(0),
        }

    def dot_attention(self, q, k, v):
        """
        q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        k=v: [bs, length, dim] or [bs, poly_m, dim]
        """
        attn_weights = torch.matmul(q, k.transpose(2, 1))  # [bs, poly_m, length]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)  # [bs, poly_m, dim]
        return output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        context_input_ids=None,
        context_attention_mask=None,
    ):
        # 쿼리 임베딩만 계산하는 경우
        if input_ids is not None and context_input_ids is None:
            query_output = self.bert(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            batch_size = input_ids.size(0)

            # Poly-code embeddings 생성
            poly_code_ids = torch.arange(self.poly_m).to(self.device)
            poly_code_ids = poly_code_ids.unsqueeze(0).expand(
                batch_size, self.poly_m
            )  # [bs, poly_m]
            poly_codes = self.poly_code_embeddings(poly_code_ids)  # [bs, poly_m, dim]

            # Dot attention 적용
            query_emb = self.dot_attention(poly_codes, query_output, query_output)
            return query_emb

        # 문맥 임베딩만 계산하는 경우
        elif context_input_ids is not None and input_ids is None:
            context_output = self.bert(
                input_ids=context_input_ids, attention_mask=context_attention_mask
            ).last_hidden_state
            context_emb = context_output[:, 0, :]  # [CLS] 토큰 사용
            return context_emb

        # 쿼리와 문맥 임베딩 모두가 주어진 경우 (정규 forward, 점수 계산)
        elif input_ids is not None and context_input_ids is not None:
            query_output = self.bert(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            batch_size = input_ids.size(0)

            # Poly-code embeddings 생성
            poly_code_ids = torch.arange(self.poly_m).to(self.device)
            poly_code_ids = poly_code_ids.unsqueeze(0).expand(
                batch_size, self.poly_m
            )  # [bs, poly_m]
            poly_codes = self.poly_code_embeddings(poly_code_ids)  # [bs, poly_m, dim]

            # Dot attention 적용
            query_emb = self.dot_attention(poly_codes, query_output, query_output)

            # 문맥 임베딩 계산
            context_output = self.bert(
                input_ids=context_input_ids, attention_mask=context_attention_mask
            ).last_hidden_state
            context_emb = context_output[:, 0, :]  # [CLS] 토큰

            # 쿼리와 문맥 임베딩 간의 점수 계산
            scores = torch.matmul(query_emb, context_emb.unsqueeze(-1)).squeeze(
                -1
            )  # [bs, poly_m]
            return query_emb, context_emb, scores


class DenseRetrievalPolyEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path,
        data_path,
        context_path,
        poly_m=64,
        margin=2,
        device="cuda",
    ):
        super(DenseRetrievalPolyEncoder, self).__init__()
        self.model = PolyEncoder(model_name_or_path, poly_m=poly_m, device=device).to(
            device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.loss_fct = nn.TripletMarginLoss(margin=margin)  # Triplet Margin Loss 적용
        self.data_path = data_path
        self.context_path = context_path
        self.device = device
        self.faiss_index = None

        # Load Wikipedia data (contexts) from json file
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # Extract 'text' field from wiki and remove duplicates
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

    def build_faiss_index(self, context_embeddings):
        # L2 거리를 기반으로 faiss 인덱스 생성하기
        dim = context_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)

        # FAISS 인덱스에 문맥 추가하기
        context_embeddings_np = context_embeddings.cpu().numpy()
        self.faiss_index.add(context_embeddings_np)
        print(
            f"FAISS 인덱스가 {len(context_embeddings_np)} 개의 문맥 임베딩으로 빌드되었습니다."
        )

    def encode_query_context(self, query, context):
        query_inputs = self.tokenizer(
            query, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        context_inputs = self.tokenizer(
            context, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        score = self.model(
            query_inputs.input_ids,
            query_inputs.attention_mask,
            context_inputs.input_ids,
            context_inputs.attention_mask,
        )
        return score.item()

    def encode_query_context_batch(self, queries):
        query_embeddings = []  # 쿼리 임베딩 저장소

        # 쿼리와 문맥 각각에 대해 임베딩 계산
        for query in queries:
            # 쿼리를 토크나이징 후 모델을 통해 임베딩 계산
            query_inputs = self.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                query_embedding = self.model(
                    input_ids=query_inputs["input_ids"],
                    attention_mask=query_inputs["attention_mask"],
                )

            query_embeddings.append(query_embedding)

        # 캐시된 문맥 임베딩 불러오기
        context_embeddings = self.cached_context_embeddings

        return query_embeddings, context_embeddings

    def retrieve(self, query_or_dataset, topk: int = 5, query_batch_size: int = 8):
        print("retrieve를 진행한다 - DRPE")
        total_results = []

        # GPU 사용 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 단일 쿼리 처리
        if isinstance(query_or_dataset, str):
            print("단일 쿼리 처리를 진행한다.")
            scores = []
            context_batch_size = 16

            for start_idx in tqdm(
                range(0, len(self.contexts), context_batch_size),
                desc="단일 쿼리 컨텍스트 처리",
            ):
                context_batch = self.contexts[
                    start_idx : start_idx + context_batch_size
                ]
                batch_scores = self.encode_query_context_batch(
                    [query_or_dataset] * len(context_batch)
                )

                # batch_scores가 임베딩 벡터일 경우, 유사도를 사용해 단일 값으로 변환
                for query_embedding, context_embedding in zip(
                    batch_scores[0], batch_scores[1]
                ):
                    query_tensor = torch.tensor(
                        query_embedding, dtype=torch.float32
                    ).to(
                        device
                    )  # float32로 변환 후 GPU로 이동
                    context_tensor = torch.tensor(
                        context_embedding, dtype=torch.float32
                    ).to(
                        device
                    )  # float32로 변환 후 GPU로 이동

                    # 코사인 유사도 계산
                    similarity = F.cosine_similarity(
                        query_tensor, context_tensor, dim=-1
                    )
                    scores.append(similarity.mean().item())  # 단일 값으로 변환 후 저장

            # 상위 k개 문서 선택
            topk_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:topk]
            topk_contexts = [self.contexts[i] for i in topk_indices]
            topk_scores = [scores[i] for i in topk_indices]

            return pd.DataFrame(
                {
                    "query": query_or_dataset,
                    "topk_contexts": topk_contexts,
                    "topk_scores": topk_scores,
                }
            )

        # 다중 쿼리 처리 (PolyEncoder 방식으로 개선)
        elif isinstance(query_or_dataset, Dataset):
            print("다중 쿼리 처리를 진행한다.")
            queries = query_or_dataset["question"]

            # 쿼리 배치와 문서 배치를 동시에 인코딩하고 PolyEncoder를 사용하여 처리
            for start_idx in tqdm(
                range(0, len(queries), query_batch_size), desc="Processing queries"
            ):
                query_batch = queries[start_idx : start_idx + query_batch_size]

                # 쿼리와 문서 배치를 한번에 인코딩
                query_inputs = self.tokenizer(
                    query_batch, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                context_inputs = self.tokenizer(
                    self.contexts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                # PolyEncoder로 쿼리 배치와 문서 배치에 대한 점수 계산
                scores = self.model(
                    query_inputs.input_ids,
                    query_inputs.attention_mask,
                    context_inputs.input_ids,
                    context_inputs.attention_mask,
                )

                # 상위 k개의 문서 선택
                topk_indices = torch.topk(scores, topk, dim=1).indices
                for i, query in tqdm(
                    enumerate(query_batch), desc="Processing documents"
                ):
                    topk_contexts = [self.contexts[idx] for idx in topk_indices[i]]
                    topk_scores = scores[i][topk_indices[i]].tolist()

                    total_results.append(
                        {
                            "query": query,
                            "topk_contexts": topk_contexts,
                            "topk_scores": topk_scores,
                        }
                    )

            return pd.DataFrame(total_results)

    def preprocess_function(self, examples, tokenizer):
        # 쿼리와 문맥에 대해 각각 토큰화 진행
        query_encodings = tokenizer(examples["question"], truncation=True, padding=True)

        # positive_context는 실제 context를 사용
        positive_encodings = tokenizer(
            examples["context"], truncation=True, padding=True
        )

        # negative_context는 다른 문서를 무작위로 선택 (배치 내에서 무작위로 섞기)
        shuffled_contexts = random.sample(examples["context"], len(examples["context"]))
        negative_encodings = tokenizer(shuffled_contexts, truncation=True, padding=True)

        return {
            "query_input_ids": query_encodings["input_ids"],
            "query_attention_mask": query_encodings["attention_mask"],
            "positive_context_input_ids": positive_encodings["input_ids"],
            "positive_context_attention_mask": positive_encodings["attention_mask"],
            "negative_context_input_ids": negative_encodings["input_ids"],
            "negative_context_attention_mask": negative_encodings["attention_mask"],
        }

    def custom_collate_fn(self, batch):
        # 각 요소에서 'input_ids'와 'attention_mask'를 추출하여 패딩
        query_input_ids = [torch.tensor(b["query_input_ids"]) for b in batch]
        query_attention_mask = [torch.tensor(b["query_attention_mask"]) for b in batch]

        positive_context_input_ids = [
            torch.tensor(b["positive_context_input_ids"]) for b in batch
        ]
        positive_context_attention_mask = [
            torch.tensor(b["positive_context_attention_mask"]) for b in batch
        ]

        negative_context_input_ids = [
            torch.tensor(b["negative_context_input_ids"]) for b in batch
        ]
        negative_context_attention_mask = [
            torch.tensor(b["negative_context_attention_mask"]) for b in batch
        ]

        # 패딩을 통해 각 시퀀스의 길이를 맞춤
        padded_query_input_ids = pad_sequence(
            query_input_ids, batch_first=True, padding_value=0
        )
        padded_query_attention_mask = pad_sequence(
            query_attention_mask, batch_first=True, padding_value=0
        )

        padded_positive_context_input_ids = pad_sequence(
            positive_context_input_ids, batch_first=True, padding_value=0
        )
        padded_positive_context_attention_mask = pad_sequence(
            positive_context_attention_mask, batch_first=True, padding_value=0
        )

        padded_negative_context_input_ids = pad_sequence(
            negative_context_input_ids, batch_first=True, padding_value=0
        )
        padded_negative_context_attention_mask = pad_sequence(
            negative_context_attention_mask, batch_first=True, padding_value=0
        )

        return {
            "query_input_ids": padded_query_input_ids,
            "query_attention_mask": padded_query_attention_mask,
            "positive_context_input_ids": padded_positive_context_input_ids,
            "positive_context_attention_mask": padded_positive_context_attention_mask,
            "negative_context_input_ids": padded_negative_context_input_ids,
            "negative_context_attention_mask": padded_negative_context_attention_mask,
        }

    def hard_negative_mining(self, query_emb, negative_embs, positive_emb, margin=1.5):
        """
        Hard Negative Mining: Negative 샘플 중에서 Positive와 가장 가까운 Negative 샘플을 선택합니다.
        """
        query_emb_mean = query_emb.mean(dim=1)
        # Positive와 Negative 간의 거리를 계산
        negative_distances = torch.norm(
            query_emb_mean - negative_embs, dim=1
        )  # 각 Negative 샘플과의 거리

        # 가장 가까운 Negative 샘플을 선택
        hardest_negative_idx = torch.argmin(
            negative_distances
        )  # 가장 가까운 Negative 샘플 인덱스
        hard_negative = negative_embs[hardest_negative_idx]  # Hard Negative 샘플 선택

        hard_negative = hard_negative.unsqueeze(0)
        return hard_negative

    def train_step(self, batch, optimizer, scheduler):
        self.model.train()
        total_loss = 0

        query_input_ids = batch["query_input_ids"].to(self.device)
        query_attention_mask = batch["query_attention_mask"].to(self.device)
        positive_context_input_ids = batch["positive_context_input_ids"].to(
            self.device
        )  # 정답 문서
        positive_context_attention_mask = batch["positive_context_attention_mask"].to(
            self.device
        )
        negative_context_input_ids = batch["negative_context_input_ids"].to(
            self.device
        )  # 오답 문서
        negative_context_attention_mask = batch["negative_context_attention_mask"].to(
            self.device
        )

        # 쿼리와 문서 임베딩 계산
        query_emb, positive_emb, scores = self.model(
            query_input_ids,
            query_attention_mask,
            positive_context_input_ids,
            positive_context_attention_mask,
        )
        _, negative_embs, _ = self.model(
            query_input_ids,
            query_attention_mask,
            negative_context_input_ids,
            negative_context_attention_mask,
        )

        # Hard Negative Mining 적용
        hard_negative = self.hard_negative_mining(
            query_emb, negative_embs, positive_emb
        )

        # 임베딩 차원이 2D인지 확인 후 연산
        query_emb_mean = query_emb.mean(dim=1)
        positive_emb_mean = positive_emb

        # TripletMarginLoss로 손실 계산
        loss = self.loss_fct(
            query_emb_mean, positive_emb_mean, hard_negative
        )  # 쿼리와 정답 문서 임베딩 간의 거리는 최소화하고, 오답 문서와는 최대화

        # 스코어 값 출력
        if random.random() < 0.05:  # 5% 확률로 스코어 확인
            positive_score = torch.norm(query_emb_mean - positive_emb_mean, p=2)
            negative_score = torch.norm(query_emb_mean - hard_negative, p=2)
            print(f"Positive Score (L2 distance): {positive_score}")
            print(f"Negative Score (L2 distance): {negative_score}")
            print(f"Loss: {loss.item()}")

        # 역전파 및 최적화
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        return total_loss

    def train_model(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        epochs=1,
        lr=5e-6,
        warmup_steps=500,
    ):
        # Optimizer와 Scheduler 설정
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # 학습 시작
        for epoch in tqdm(range(epochs), desc="에폭: "):
            self.model.train()  # 모델을 학습 모드로 설정
            total_loss = 0

            print(f"Epoch {epoch + 1}/{epochs}")

            print("Length of Train DataLoader: ", len(train_dataloader))

            for step, batch in tqdm(
                enumerate(train_dataloader), desc="데이터 트레이닝하기"
            ):
                optimizer.zero_grad()  # 이전 배치의 gradient 초기화

                # 각 배치에서 train_step 호출하여 손실 계산 및 학습
                loss = self.train_step(batch, optimizer, scheduler)  # 배치 단위 학습
                total_loss += loss

                if step % 100 == 0:  # 100 스텝마다 현재 손실 출력
                    print(f"Step {step}, Loss: {loss}")

    def cache_context_embeddings(self, batch_size=16):  # 배치 크기 조정 가능
        context_embeddings = []
        for start_idx in tqdm(
            range(0, len(self.contexts), batch_size), desc="문맥 임베딩 계산"
        ):
            context_batch = self.contexts[start_idx : start_idx + batch_size]
            context_inputs = self.tokenizer(
                context_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():  # 메모리 절약을 위해 no_grad 사용
                _, context_embedding, _ = self.model(
                    input_ids=context_inputs.input_ids,
                    attention_mask=context_inputs.attention_mask,
                    context_input_ids=context_inputs.input_ids,
                    context_attention_mask=context_inputs.attention_mask,
                )
            context_embeddings.append(context_embedding.cpu())

        # 캐시 파일로 저장
        torch.save(torch.cat(context_embeddings), "cached_context_embeddings.pt")

    def load_cached_context_embeddings(self):
        self.cached_context_embeddings = torch.load("cached_context_embeddings.pt").to(
            self.device
        )


# Dense 관련 코드
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poly Encoder-based Dense Retrieval")
    parser.add_argument(
        "--dataset_name",
        metavar="./data/train_dataset",
        type=str,
        help="Dataset path",
        default="../data/train_dataset",
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="klue/bert-base",
        type=str,
        help="Model name or path for PolyEncoder",
        default="klue/bert-base",
    )
    parser.add_argument(
        "--data_path",
        metavar="./data",
        type=str,
        help="Path to data",
        default="../data",
    )
    parser.add_argument(
        "--context_path",
        metavar="wikipedia_documents",
        type=str,
        help="Path to contexts (wikipedia data)",
        default="wikipedia_documents.json",
    )
    parser.add_argument(
        "--use_faiss",
        metavar=True,
        type=bool,
        help="Use FAISS for retrieval",
        default=False,
    )

    args = parser.parse_args()

    # Test dataset load
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )
    print("*" * 40, "query dataset", "*" * 40)
    torch.cuda.empty_cache()
    # Initialize PolyEncoder-based Dense Retrieval
    retriever = DenseRetrievalPolyEncoder(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        context_path=args.context_path,
        poly_m=64,  # You can adjust this value based on your needs
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:
        # FAISS-based retrieval
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # Bulk query processing
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)
            print(f"Top results for query: {query}")
    else:
        # Single query evaluation
        with timer("single query by exhaustive search"):
            if os.path.exists("cached_context_embeddings.pt"):
                print("캐시된 문맥 임베딩을 불러옵니다.")
                retriever.load_cached_context_embeddings()
            else:
                print("문맥 임베딩을 계산하고 캐싱합니다.")
                retriever.cache_context_embeddings()

            result = retriever.retrieve(query)
            print(result)

        #    Exhaustive DPR Search using PolyEncoder
        with timer("bulk query by exhaustive search"):

            def show_differences(str1, str2):
                print("Original Context:\n", repr(str1))
                print("Corrected Context:\n", repr(str2))
                if str1 != str2:
                    for i, (a, b) in enumerate(zip(str1, str2)):
                        if a != b:
                            print(f"Difference at index {i}: '{a}' != '{b}'")

            # PolyEncoder-based retrieval for multiple queries
            df = retriever.retrieve(full_ds, topk=40, query_batch_size=4)
            print(df)

            # 데이터프레임의 열 목록을 출력
            print("df columns: ", df.columns)

            if (
                df is not None
                and "original_context" in df.columns
                and "first_context" in df.columns
            ):
                df["correct"] = df["original_context"] == df["first_context"]
                print(
                    "correct retrieval result by exhaustive search",
                    df["correct"].sum() / len(df),
                )
            else:
                print("Error: DataFrame doesn't contain required columns")

# BM25 관련 코드
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="")
#     parser.add_argument(
#         "--dataset_name", metavar="./data/train_dataset", type=str, help="", default="../data/train_dataset"
#     )
#     parser.add_argument(
#         "--model_name_or_path",
#         metavar="klue/roberta-large",
#         type=str,
#         help="",
#         default="klue/roberta-large",
#     )
#     parser.add_argument("--data_path", metavar="./data", type=str, help="", default="../data")
#     parser.add_argument(
#         "--context_path", metavar="wikipedia_documents", type=str, help="", default="wikipedia_documents.json"
#     )
#     parser.add_argument("--use_faiss", metavar=False, type=bool, help="", default=False)

#     args = parser.parse_args()

#     # Test sparse
#     org_dataset = load_from_disk(args.dataset_name)
#     full_ds = concatenate_datasets(
#         [
#             org_dataset["train"].flatten_indices(),
#             org_dataset["validation"].flatten_indices(),
#         ]
#     )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
#     print("*" * 40, "query dataset", "*" * 40)
#     print(full_ds)

#     # Kiwi 초기화
#     kiwi = Kiwi()

#     # tokenize_fn 정의
#     def kiwipiepy_tokenize(text):
#         try:
#             cleaned_text = text.encode('utf-8', 'ignore').decode('utf-8')
#             tokens = kiwi.tokenize(cleaned_text)
#             return [token.form for token in tokens]
#         except (UnicodeDecodeError, AttributeError) as e:
#             print(f"유니코드 디코딩 오류 발생, 지문 건너뛰기: {e}")
#             # 오류 발생 시 빈 리스트를 반환하여 해당 지문을 무시
#             return []
#         except Exception as e:
#             # 예상치 못한 다른 에러 발생 시 처리
#             print(f"알 수 없는 오류 발생, 지문 건너뛰기: {e}")
#             return []

#     retriever = SparseRetrieval(
#         tokenize_fn=kiwipiepy_tokenize,
#         data_path=args.data_path,
#         context_path=args.context_path,
#     )

#     query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

#     if args.use_faiss:

#         # test single query
#         with timer("single query by faiss"):
#             scores, indices = retriever.retrieve_faiss(query)

#         # test bulk
#         with timer("bulk query by exhaustive search"):
#             retriever.get_sparse_embedding()
#             df = retriever.retrieve_faiss(full_ds)
#             df["correct"] = df["original_context"] == df["context"]


#             print("correct retrieval result by faiss", df["correct"].sum() / len(df))

# else:
#     with timer("bulk query by exhaustive search"):
#         def show_differences(str1, str2):
#             # 각 문자열을 줄 단위로 나눠서 보여줌
#             print("Original Context:\n", repr(str1))
#             print("Corrected Context:\n", repr(str2))

#             # 문자열이 동일하지 않을 경우 차이점을 출력
#             if str1 != str2:
#                 for i, (a, b) in enumerate(zip(str1, str2)):
#                     if a != b:
#                         print(f"Difference at index {i}: '{a}' != '{b}'")


#         df = retriever.retrieve(full_ds)
#         df["correct"] = df["original_context"] == df["first_context"]
#         # 데이터프레임의 각 행을 비교하고 차이점을 출력
#         print(
#             "correct retrieval result by exhaustive search",
#             df["correct"].sum() / len(df),
#         )

#     with timer("single query by exhaustive search"):
#         scores, indices = retriever.retrieve(query)
