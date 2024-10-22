from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import Dataset
import torch

#lcw99/t5-base-korean-paraphrase
model_name = 'KETI-AIR/ke-t5-large'

# Set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = T5ForConditionalGeneration.from_pretrained(model_name).to(device) # Move model to GPU
tokenizer = T5Tokenizer.from_pretrained(model_name)

def generate_questions(batch):
    # Prepare inputs for the entire batch
    input_texts = [f"paraphrase: {question}" for question in batch['question']]

    # for context, answer_dict in zip(batch['context'], batch['answers']):
    #     answer = answer_dict['text']
    #     input_texts.append(f"generate question: context: {context} answer: {answer}")

    
    # Tokenize the inputs
    inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Move inputs to the device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Generate questions for the entire batch
    outputs = model.generate(inputs['input_ids'], max_length=64, num_beams=4, early_stopping=True)

    # Decode the generated questions
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    return {'question': questions}


train_data = Dataset.load_from_disk('../data/train_dataset/train')
    
train_data = train_data.map(
    generate_questions,
    batched=True,
    batch_size=8
)

train_data.save_to_disk('../data/paraphrase_t5_train_dataset')