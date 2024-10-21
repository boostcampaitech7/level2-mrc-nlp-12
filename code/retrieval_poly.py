import argparse
import os
import json
import torch
import torch.nn.functional as F

from datasets import load_from_disk, concatenate_datasets
from poly_enc import PolyEncoder
from tqdm import tqdm 
from torch.optim import AdamW
from transformers import AutoTokenizer
from torch.utils.data import DataLoader 


# Hard Negative Mining -> Gold Negative 활용
def train_polyencoder(poly_encoder, train_dataloader, tokenizer, num_epochs=10, learning_rate=2e-5, device='cuda'): 
    # Optimizer 정의
    optimizer = AdamW(poly_encoder.parameters(), lr=learning_rate) 
    
    # 손실 함수 정의 (여기서는 Cosine Embedding Loss를 사용)
    criterion = torch.nn.MSELoss()
    
    # 모델을 학습 모드로 설정
    poly_encoder.train()
    
    for epoch in range(num_epochs): 
      total_loss = 0.0
      
      for batch in tqdm(train_dataloader[:100], desc=f"Training 에폭: {epoch + 1}"): 
        optimizer.zero_grad()
        
        # 배치 내의 모든 질문과 컨텍스트를 토크나이징
        questions_input = tokenizer(batch['question'], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        contexts_input = tokenizer(batch['context'], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device) 
        negative_contexts_input = contexts_input['input_ids'].roll(shifts=1, dims=0)  # 샘플로 롤링

        
        # Poly Encoder 모델을 통해 임베딩 생성
        question_embeddings = poly_encoder.cand_encoder(questions_input['input_ids'])[:, 0, :]
        context_embeddings = poly_encoder.context_encoder(contexts_input['input_ids'])[:, 0, :]
        negative_embeddings = poly_encoder.context_encoder(negative_contexts_input)[:, 0, :]
        
        # 벡터 정규화
        question_embeddings = F.normalize(question_embeddings, p=2, dim=-1)
        context_embeddings = F.normalize(context_embeddings, p=2, dim=-1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=-1)

        # Positive와 Negative 임베딩 간의 유사도 계산 
        # Shape: [batch_size, embedding_dim]
        positive_similarity = torch.sum(question_embeddings * context_embeddings, dim=-1, keepdim=True)
        
        # Negative 유사도 계산 (Dot Product)
        negative_similarity = torch.sum(question_embeddings.unsqueeze(1) * negative_embeddings, dim=-1) 
      
        combined_similarities = torch.cat(
          [positive_similarity, negative_similarity.view(-1, 1)], dim=0) 
      
        # 손실 계산을 위한 target 설정 및 손실 계산
        batch_size = positive_similarity.size(0)  # batch size를 얻음
        num_negatives = negative_similarity.size(1)  # negative samples의 개수 얻음 
        
        # 손실 계산을 위한 target 설정 및 손실 계산
        target = torch.cat([
            torch.ones(batch_size, 1, device=device),  # Positive -> 1
            torch.zeros(batch_size * num_negatives, 1, device=device)  # Negative -> 0
        ])
        
        # 손실 계산
        loss = criterion(combined_similarities, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
      
      avg_loss = total_loss / len(train_dataloader)
      print(f"Epoch {epoch + 1} - Average Loss: {avg_loss}")
      
      # 모델 저장 (매 에포크마다 저장할 수도 있음)
      save_path = f"./saved_models/poly_encoder_epoch_{epoch + 1}.pth"
      torch.save(poly_encoder.state_dict(), save_path)
      print(f"Model saved to {save_path}")
      
def load_embeddings_in_chunks(file_path, chunk_size=1000):
    """
    큰 임베딩 파일을 작은 블록 단위로 로드하는 함수.
    
    Args:
        file_path (str): 임베딩 파일 경로
        chunk_size (int): 한 번에 불러올 임베딩의 개수
    
    Returns:
        torch.Tensor: 임베딩 전체를 합친 텐서
    """
    embeddings_list = []  # 임베딩을 저장할 리스트

    # 임베딩 파일을 차례대로 읽기
    with open(file_path, 'r') as f:
        current_chunk = []

        for i, line in enumerate(f):
            embedding = list(map(float, line.strip().split()))  # 각 줄을 임베딩으로 변환
            current_chunk.append(embedding)

            # chunk_size에 도달하면 임시 저장 후 초기화
            if (i + 1) % chunk_size == 0:
                embeddings_list.append(torch.tensor(current_chunk))
                current_chunk = []

        # 남아 있는 데이터 처리
        if current_chunk:
            embeddings_list.append(torch.tensor(current_chunk))

    # 임베딩 전체를 하나의 텐서로 합치기
    all_embeddings = torch.cat(embeddings_list, dim=0)
    return all_embeddings
        
        
        
        
    
    

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
        default="klue/bert-base"
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
  
  # 저장된 모델 불러오기 (학습이 완료된 후 모델을 재사용할 때)
  load_path = "./saved_models/poly_encoder_epoch_1.pth"  # 필요한 에포크의 파일 경로

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

  tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
  
  # DataLoder를 생성한다 (배치 크기 설정)
  batch_size = 4  # 상황에 따라 최적화 필요
  corpus_dataloader = DataLoader(corpus_texts, batch_size=batch_size, shuffle=False) 
  
  # GPU/CPU 설정 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  poly_encoder.to(device)
  
  train_dataloader = DataLoader(
    org_dataset["train"].flatten_indices(), batch_size=4, shuffle=True)
  
  # 학습 함수 호출 
  load_path = "./saved_models/poly_encoder_epoch_1.pth" 
  if os.path.exists(load_path):
    poly_encoder.load_state_dict(torch.load(load_path))
    poly_encoder.eval()  # 평가 모드로 전환
    print(f"Model loaded from {load_path}")
  else: 
    train_polyencoder(poly_encoder, train_dataloader, tokenizer, num_epochs=3, learning_rate=1e-5, device='cuda')
  

  
  # 임베딩을 저장할 파일 경로 설정
  embedding_file = "context_embeddings_poly_encoder.pt"
  embedding_dir = "context_embeddings" 
  embedding_file = os.path.join(embedding_dir, "context_embeddings_poly_encoder.pt")

  # 파일이 존재하는지 확인하고 불러오기
  if os.path.exists(embedding_file):
      print("임베딩 파일을 불러옵니다...")
      # 모든 임베딩 파일을 불러와서 병합
      context_embeddings = []
      for file_name in tqdm(sorted(os.listdir(embedding_dir)), desc="임베딩 파일 불러오는 중..."):
        if file_name.startswith("context_embeddings_") and file_name.endswith(".pt"):
            file_path = os.path.join(embedding_dir, file_name)
            context_embeddings.extend(torch.load(file_path))
  else:
      print("임베딩 파일이 없으므로 새로 생성합니다...")
      # 코퍼스를 인코딩한다. 
      context_embeddings = []
      
      save_interval = 1000  # 임베딩 1000개마다 저장
      batch_count = 0
      
      for batch_text in tqdm(corpus_dataloader, desc="코퍼스 인코딩: "):
        for text in batch_text:
          # 문서의 첫 512 토큰만 인코딩
          tok_ids = tokenizer.encode(
            text, add_special_tokens=True, max_length=512, truncation=True, return_tensors="pt").to(device)

          with torch.no_grad():
            encoded_text = poly_encoder.context_encoder(tok_ids)
              
          # 인코딩 결과를 저장
          context_embeddings.append(encoded_text.cpu())
          
          # 메모리를 절약하기 위해 일정 수의 임베딩마다 저장
          batch_count += 1
          if batch_count % save_interval == 0:
            partial_file = os.path.join(embedding_dir, f"context_embeddings_{batch_count}.pt")
            torch.save(context_embeddings, partial_file)
            print(f"임베딩을 {partial_file} 파일에 부분 저장했습니다.")
            context_embeddings = []  # 메모리 해제
      
      # 마지막 남은 임베딩 파일로 저장   
      if context_embeddings:   
        torch.save(context_embeddings, embedding_file)
        print(f"임베딩을 {embedding_file} 파일에 저장했습니다.")
        
  # 질문 임베딩 생성
  question = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
  question_tokens = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512).input_ids.to(device) 
  with torch.no_grad():
      question_embedding = poly_encoder.cand_encoder(question_tokens)
      
  # 유사도 계산 (예: dot-product)
  scores = []
  for context_embedding in tqdm(context_embeddings, desc="유사도 계산: "): 
      context_embedding = context_embedding.to(device)
      
      question_vector = torch.mean(question_embedding, dim=(0, 1))
      context_vector = torch.mean(context_embedding, dim=(0, 1))

      similarity = torch.dot(question_vector, context_vector)
      scores.append(similarity.item())
      
  # 상위 n개 문서 선택
  top_k = 5
  top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k] 

  for idx in top_indices:
      print(f"Top Document {idx + 1}:") 
      print(f"Scores: {scores[idx]}")
      print(corpus_texts[idx])  # corpus_texts에서 상위 문서의 내용을 출력
      print("=" * 50)