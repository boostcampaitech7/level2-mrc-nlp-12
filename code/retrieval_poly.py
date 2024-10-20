import argparse
import os
import json
import torch 

from datasets import load_from_disk, concatenate_datasets
from poly_enc import PolyEncoder
from tqdm import tqdm 
from torch.optim import AdamW
from transformers import AutoTokenizer
from torch.utils.data import DataLoader 

# Hard Negative Mining -> Gold Negative 활용
def train_polyencoder(poly_encoder, train_dataloader, tokenizer, num_epochs=3, learning_rate=1e-5, device='cuda'): 
    # Optimizer 정의
    optimizer = AdamW(poly_encoder.parameters(), lr=learning_rate) 
    
    # 손실 함수 정의 (여기서는 Cosine Embedding Loss를 사용)
    criterion = torch.nn.CosineEmbeddingLoss()
    
    # 모델을 학습 모드로 설정
    poly_encoder.train()
    
    for epoch in range(num_epochs): 
      total_loss = 0.0
      
      for batch in tqdm(train_dataloader, desc=f"Training 에폭: {epoch + 1}"): 
        optimizer.zero_grad() 

        # 배치 데이터 가져오기
        for batch_index, question in enumerate(batch['question']):
          
          positive_context = batch['context'][batch_index]
          negative_contexts = [context for i, context in enumerate(batch['context']) if i != batch_index]
          
          # 질문 및 컨텍스트를 토크나이징
          question_input = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
          positive_input = tokenizer(positive_context, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
          negative_inputs = [tokenizer(neg, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device) for neg in negative_contexts] 
          
          # Poly Encoder 모델을 통해 임베딩 생성
          positive_embedding = poly_encoder.context_encoder(positive_input['input_ids'])
          question_embedding = poly_encoder.cand_encoder(question_input['input_ids'])
          
          # Negative embedding 생성
          negative_embeddings = [poly_encoder.context_encoder(neg_input['input_ids']) for neg_input in negative_inputs]
          
          # Positive와 Negative의 유사도 계산
          positive_similarity = torch.cosine_similarity(
              torch.mean(question_embedding, dim=1),
              torch.mean(positive_embedding, dim=1)
          )
          
          # 목표 값 설정
          negative_similarities = torch.stack([
              torch.cosine_similarity(
                  torch.mean(question_embedding, dim=1),
                  torch.mean(neg_emb, dim=1)
              ) for neg_emb in negative_embeddings
          ])
          batch_size = negative_similarities.size(0)
          target_positive = torch.ones(batch_size, device=device)  # Positive 유사도는 1이어야 함
          target_negative = -torch.ones(batch_size, device=device)  # Negative 유사도는 -1이어야 함
          positive_similarities_expanded = positive_similarity.expand_as(negative_similarities)
    
          # 손실 계산 (positive_similarity는 1로, negative_similarities는 -1로 레이블링)
          loss_positive = criterion(positive_similarities_expanded, negative_similarities, target_positive)
          loss_negative = criterion(negative_similarities, negative_similarities, target_negative)

          # 최종 손실
          loss = (loss_positive + loss_negative) / 2
          loss.backward()
          optimizer.step()

          total_loss += loss.item()
          
        avg_loss = total_loss / len(train_dataloader)
        print(f"Positive similarities: {positive_similarity}")
        print(f"Negative similarities: {negative_similarities}")
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss}")
        
        
        
        
    
    

if __name__ == "__main__": 
  parser = argparse.ArgumentParser(description="Poly Encoder-based Dense Retrieval")
  parser.add_argument(
        "--dataset_name", metavar="./data/train_dataset", type=str, help="Dataset path", default="../data/train_dataset"
  )    
  parser.add_argument(
        "--model_name_or_path",
        metavar="klue/roberta-large",
        type=str,
        help="Model name or path for PolyEncoder", 
        default="klue/roberta-large",
  )
  parser.add_argument("--data_path", metavar="./data", type=str, help="Path to data", default="../data")
  parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, help="Path to contexts (wikipedia data)", default="wikipedia_documents.json"
  )
  parser.add_argument("--use_faiss", metavar=True, type=bool, help="Use FAISS for retrieval", default=False)
  parser.add_argument("--embed_size", metavar=1024, type=bool, help="Use FAISS for retrieval", default=1024)

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
  poly_encoder = PolyEncoder(args, model_name="klue/roberta-large", pooling_method="first", n_codes=64)

  # 토크나이저를 준비한다. -> 시스템 제한 에러가 발생해서 일단 보류함
  # def sliding_window_tokenizer(text, tokenizer, max_length=512, stride=256):
  #   input_ids = tokenizer.encode(text, add_special_tokens=True) 

  #   # 슬라이딩 윈도우 기법으로 자르기 
  #   chunks = [] 
  #   for i in range(0, len(input_ids), stride): 
  #     chunk = input_ids[i:i + max_length] 
  #     if len(chunk) < max_length: # 마지막 chunk가 max_length보다 작으면, padding을 추가
  #       chunk += [tokenizer.pad_token_id] * (max_length - len(chunk))
  #     chunks.append(torch.tensor(chunk)) 

  #   return chunks

  tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
  
  # DataLoder를 생성한다 (배치 크기 설정)
  batch_size = 4  # 상황에 따라 최적화 필요
  corpus_dataloader = DataLoader(corpus_texts, batch_size=batch_size, shuffle=False) 
  
  # GPU/CPU 설정 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  poly_encoder.to(device)
  
  train_dataloader = DataLoader(
    org_dataset["train"].flatten_indices(), batch_size=8, shuffle=True)
  optimizer = torch.optim.Adam(poly_encoder.parameters(), lr=1e-5)
  
  # 학습 함수 호출
  train_polyencoder(poly_encoder, train_dataloader, tokenizer, num_epochs=3, learning_rate=1e-5, device='cuda')
  

  
  # 임베딩을 저장할 파일 경로 설정
  embedding_file = "context_embeddings_poly_encoder.pt"
  embedding_dir = "context_embeddings" 
  embedding_file = os.path.join(embedding_dir, "context_embeddings_poly_encoder.pt")

  # # 파일이 존재하는지 확인하고 불러오기
  # if os.path.exists(embedding_file):
  #     print("임베딩 파일을 불러옵니다...")
  #     # 모든 임베딩 파일을 불러와서 병합
  #     context_embeddings = []
  #     for file_name in tqdm(sorted(os.listdir(embedding_dir)), desc="임베딩 파일 불러오는 중..."):
  #       if file_name.startswith("context_embeddings_") and file_name.endswith(".pt"):
  #           file_path = os.path.join(embedding_dir, file_name)
  #           context_embeddings.extend(torch.load(file_path))
  # else:
  #     print("임베딩 파일이 없으므로 새로 생성합니다...")
  #     # 코퍼스를 인코딩한다. 
  #     context_embeddings = []
      
  #     save_interval = 1000  # 임베딩 1000개마다 저장
  #     batch_count = 0
      
  #     for batch_text in tqdm(corpus_dataloader, desc="코퍼스 인코딩: "):
  #       for text in batch_text:
  #         # 문서의 첫 512 토큰만 인코딩
  #         tok_ids = tokenizer.encode(
  #           text, add_special_tokens=True, max_length=512, truncation=True, return_tensors="pt").to(device)

  #         with torch.no_grad():
  #           encoded_text = poly_encoder.context_encoder(tok_ids)
              
  #         # 인코딩 결과를 저장
  #         context_embeddings.append(encoded_text.cpu())
          
  #         # 메모리를 절약하기 위해 일정 수의 임베딩마다 저장
  #         batch_count += 1
  #         if batch_count % save_interval == 0:
  #           partial_file = os.path.join(embedding_dir, f"context_embeddings_{batch_count}.pt")
  #           torch.save(context_embeddings, partial_file)
  #           print(f"임베딩을 {partial_file} 파일에 부분 저장했습니다.")
  #           context_embeddings = []  # 메모리 해제
      
  #     # 마지막 남은 임베딩 파일로 저장   
  #     if context_embeddings:   
  #       torch.save(context_embeddings, embedding_file)
  #       print(f"임베딩을 {embedding_file} 파일에 저장했습니다.")
        
  # # 질문 임베딩 생성
  # question = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
  # question_tokens = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512).input_ids.to(device) 
  # with torch.no_grad():
  #     question_embedding = poly_encoder.cand_encoder(question_tokens)
      
  # # 유사도 계산 (예: dot-product)
  # scores = []
  # for context_embedding in tqdm(context_embeddings, desc="유사도 계산: "): 
  #     context_embedding = context_embedding.to(device)
      
  #     question_vector = torch.mean(question_embedding, dim=(0, 1))
  #     context_vector = torch.mean(context_embedding, dim=(0, 1))

  #     similarity = torch.dot(question_vector, context_vector)
  #     scores.append(similarity.item())
      
  # # 상위 n개 문서 선택
  # top_k = 5
  # top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k] 

  # for idx in top_indices:
  #     print(f"Top Document {idx + 1}:") 
  #     print(f"Scores: {scores[idx]}")
  #     print(corpus_texts[idx])  # corpus_texts에서 상위 문서의 내용을 출력
  #     print("=" * 50)