from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time


def enable_gpu(model, device):
    return model.to(device)


def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
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
    model_inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)

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


if __name__ == '__main__':
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

    # Enable GPU
    model = enable_gpu(model, device)

    gen_kwargs = {
        "max_length": 512,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 3,
    }

    # Text to extract triplets from
    ''' SMALL TEXT '''
    # text = 'Punta Cana is a resort town in the municipality of Higüey, in La Altagracia Province, the easternmost province of the Dominican Republic.'
    ''' LONG TEXT '''
    # from scripts.get_text_from_pdf import extract_text_from_pdf
    # text = extract_text_from_pdf('./Knowledge Graph/wp_202320_.pdf', skip_pages=2, end_pages=15)
    ''' MEDIUM TEXT '''
    text = '''
    Title: The Keynesian Economic Model - A Framework for Economic Stabilization
    Introduction:
    The Keynesian Economic Model, developed by the British economist John Maynard Keynes in the 1930s, stands as a cornerstone in modern economic theory and policy. At its core, this model advocates for government intervention in the economy to stabilize it during times of economic turbulence. It emerged as a response to the Great Depression, offering a fresh perspective on how to address economic downturns and reduce unemployment.
    Key Principles:
    The Keynesian Model revolves around several key principles:
    Government Intervention: One of the central tenets of Keynesian economics is that in a recession or economic downturn, governments should step in and increase their spending. This boost in government expenditure helps stimulate demand in the economy, thus countering the decline in aggregate demand during a recession.
    Counter-Cyclical Policies: Keynesian economics promotes counter-cyclical policies, meaning that governments should run budget deficits during recessions and surpluses during economic booms. By doing so, they can help smooth out the economic cycle.
    The Multiplier Effect: Keynesian theory emphasizes the multiplier effect, which posits that an initial increase in government spending will result in a more significant overall increase in economic output. This is because the income generated from government spending circulates through the economy, creating a ripple effect.
    Applications:
    The Keynesian Model has been instrumental in shaping economic policy and responses to economic crises. For instance:
    Great Depression: In the 1930s, the Keynesian model influenced government policies worldwide, with many governments increasing public spending to combat the economic devastation caused by the Great Depression.
    2008 Financial Crisis: During the 2008 financial crisis, many governments adopted Keynesian policies by implementing stimulus packages to stabilize their economies.
    COVID-19 Pandemic: The COVID-19 pandemic triggered another wave of Keynesian-style stimulus measures, with governments providing financial support to individuals and businesses to mitigate the economic impact of lockdowns.
    Conclusion:
    The Keynesian Economic Model provides a crucial framework for addressing economic downturns and stabilizing economies through government intervention. By advocating for fiscal and monetary policies that actively manage demand and aggregate economic activity, this model has proven to be a valuable tool for policymakers facing economic crises. While it has its critics and limitations, the Keynesian Model remains an enduring and adaptable approach in the field of economics, offering a valuable perspective on the management of modern economies.
    '''

    start = time.time()
    triplets = generate_triplets(text, model, tokenizer, gen_kwargs, device)
    end = time.time()
    for idx, triplet in enumerate(triplets):
        print(f'Triplet {idx}: {triplet}')
    print(f'Elapsed time: {end - start} seconds')
