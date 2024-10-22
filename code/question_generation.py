from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import Dataset

model_name = 'KETI-AIR/ke-t5-small'

model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def generate_question(context, answer):
    # Format the input to the model
    input_text = f"generate question: context: {context} answer: {answer}"
    
    # Tokenize the input
    inputs = tokenizer.encode(input_text, return_tensors='pt')

    # Generate question using the model
    outputs = model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)

    # Decode the generated question
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question


train_data = Dataset.load_from_disk('../data/train_dataset/train')

# for i in range(len(train_data)):
#     context = train_data[i]['context']
#     answer_text = train_data[i]['answers']['text']
#     new_question = generate_question(context, answer_text)
    
train_data = train_data.map(
    lambda row, idx: {'question': generate_question(row['context'], row['answers']['text'])},
    with_indices=True
)

train_data.save_to_disk('../data/augment_t5_train_dataset')