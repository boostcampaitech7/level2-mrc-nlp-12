from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset
import torch

#KETI-AIR/ke-t5-large
model_name = 'lcw99/t5-base-korean-paraphrase'

# Set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device) # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_questions(batch):
    # Prepare inputs for the entire batch
    input_texts = [f"paraphrase: {question}" for question in batch['question']]
  
    # Tokenize the inputs
    inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True).to(device)
    
    # Generate paraphrased questions
    with torch.no_grad():  # Disable gradient calculations for efficiency
        outputs = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)

    # Decode the generated paraphrases
    paraphrased_questions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return {'question': paraphrased_questions}  # Return the new questions



train_data = Dataset.load_from_disk('../data/train_dataset/train')
    
train_data = train_data.map(
    generate_questions,
    batched=True,
    batch_size=8
)

train_data.save_to_disk('../data/paraphrase_lcw99t5_train_dataset')