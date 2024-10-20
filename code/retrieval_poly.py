import argparse
import os
import json
import torch

from datasets import load_from_disk, concatenate_datasets
from poly_enc import PolyEncoder
from tqdm import tqdm
from transformers import BertTokenizer

if __name__ == "__main__": 
  parser = argparse.ArgumentParser(description="Poly Encoder-based Dense Retrieval")
  parser.add_argument(
        "--dataset_name", metavar="./data/train_dataset", type=str, help="Dataset path", default="../data/train_dataset"
  )    
  parser.add_argument(
        "--model_name_or_path",
        metavar="klue/bert-base",
        type=str,
        help="Model name or path for PolyEncoder", 
        default="klue/bert-base",
  )
  parser.add_argument("--data_path", metavar="./data", type=str, help="Path to data", default="../data")
  parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, help="Path to contexts (wikipedia data)", default="wikipedia_documents.json"
  )
  parser.add_argument("--use_faiss", metavar=True, type=bool, help="Use FAISS for retrieval", default=False)
  parser.add_argument("--embed_size", metavar=768, type=bool, help="Use FAISS for retrieval", default=768)

  args = parser.parse_args()

  # 위키피디아 문서를 로드한다.
  with open(os.path.join(args.data_path, args.context_path), "r", encoding="utf-8") as f:
    wiki = json.load(f)
  corpus_texts = [doc['text'] for doc in wiki.values()]

  # 데이터셋을 로드한다.
  org_dataset = load_from_disk(args.dataset_name)
  full_ds = concatenate_datasets(
      [
          org_dataset["train"].flatten_indices(),
          org_dataset["validation"].flatten_indices(),
      ]
  )

  # 인스턴스를 생성한다. 
  poly_encoder = PolyEncoder(args, model_name="klue/bert-base", pooling_method="first", n_codes=64)

  # 토크나이저를 준비한다.
  def sliding_window_tokenizer(text, tokenizer, max_length=512, stride=256):
    input_ids = tokenizer.encode(text, add_special_tokens=True) 

    # 슬라이딩 윈도우 기법으로 자르기 
    chunks = [] 
    for i in range(0, len(input_ids), stride): 
      chunk = input_ids[i:i + max_length] 
      if len(chunk) < max_length: # 마지막 chunk가 max_length보다 작으면, padding을 추가
        chunk += [tokenizer.pad_token_id] * (max_length - len(chunk))
      chunks.append(torch.tensor(chunk)) 

    return chunks

  tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

  # 코퍼스를 인코딩한다. 
  context_embeddings = [] 
  for text in tqdm(corpus_texts, desc="코퍼스 인코딩: "):
    # text를 토크나이징한다
    # tok_ids = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)["input_ids"].squeeze(0)
    chunks = sliding_window_tokenizer(text, tokenizer)
    chunk_embeddings = []

    for chunk in chunks: 
      tok_ids = chunk.unsqueeze(0) # 배치 차원을 추가 
      encoded_chunk = poly_encoder.context_encoder(tok_ids)

      # 인코딩하고 결과를 저장한다 (배치 단위로 처리하는 것도 고려)
      chunk_embeddings.append(encoded_chunk)
      
    context_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0) 
    context_embeddings.append(context_embedding)