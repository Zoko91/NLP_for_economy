from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time
from scripts.get_text_from_pdf import extract_text_from_pdf


def enable_gpu(model, device):
    return model.to(device)


def extract_triplets(text):
    triplets = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets


def generate_triplets(text, model, tokenizer, gen_kwargs, device):
    # Tokenize text
    model_inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)

    # Generate
    generated_tokens = model.generate(
        model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        **gen_kwargs,
    )

    # Extract text
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    triplets = []
    for idx, sentence in enumerate(decoded_preds):
        triplets.extend(extract_triplets(sentence))

    return triplets


def process_long_text(text, model, tokenizer, gen_kwargs, device):
    max_length = gen_kwargs["max_length"]
    triplets = []

    # Split the text into segments
    segments = [text[i:i+max_length] for i in range(0, len(text), max_length)]

    for segment in segments:
        triplets.extend(generate_triplets(segment, model, tokenizer, gen_kwargs, device))

    return triplets


if __name__ == '__main__':
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    # NOTE: use Babelscape/mrebel-large for multilingual content
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

    # Enable GPU
    model = enable_gpu(model, device)

    # Tune hyperparameters
    gen_kwargs = {
        "max_length": 4096,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 3,
    }

    # Extract text from PDF file
    text = extract_text_from_pdf('./Knowledge Graph/wp_202320_.pdf')

    # Generate triplets and measure elapsed time
    start = time.time()
    triplets = process_long_text(text, model, tokenizer, gen_kwargs, device)
    end = time.time()

    # Print results
    for idx, triplet in enumerate(triplets):
        print(f'Triplet {idx}: {triplet}')
    print(f'Elapsed time: {end - start} seconds')
