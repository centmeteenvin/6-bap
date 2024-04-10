# import torch, sys, os
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import argparse, logging, re, torch
from pypdf import PdfReader
from transformers import AutoTokenizer, tokenization_utils_base, TrainingArguments, AutoModelForCausalLM, Trainer, BitsAndBytesConfig

logger = logging.getLogger(__name__)
name = "google/gemma-2b"
access = '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'
tokenizer = AutoTokenizer.from_pretrained(name , token = access)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id
trainingArgs = TrainingArguments(output_dir="trainer")        # print("Setting up quantitation")
# compute_dtype = getattr(torch, "float16")
# bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=compute_dtype,
#         bnb_4bit_use_double_quant=True,
# )

model = AutoModelForCausalLM.from_pretrained(
    name, 
    # quantization_config = bnb_config, 
    device_map = 'cuda',
    token = access
    )


# class SuppressOutput:
#     def blockPrint():
#         sys.stdout = open(os.devnull, 'w')
#     # Restore
#     def enablePrint():
#         sys.stdout = sys.__stdout__
#     def __enter__(self):
#         SuppressOutput.blockPrint()
        
#     def __exit__(self, type, value, traceback):
#         SuppressOutput.enablePrint()

# class Formatter:
#     def __init__(self, max_prompts):
#         self.max_prompts= max_prompts
    
#     def header(self, iteration: int) -> str:
#         return f"================================================================ {iteration} / {self.max_prompts}"
        
#     def footer(self, iteration: int) -> str:
#         return f"================================================================"
    
#     def prompt(self, iteration: int) -> str:
#         return "> " 
        
#     def welcome(self) -> None:
#         return f"""
# Welcome to the mistral pretrained chatbot. Feel free to ask me anything
#             """
    
# class Chatbot:
#     def __init__(self, max_prompts, formatter: Formatter):
#         self.formatter = formatter
#         self.max_prompts = max_prompts
#         self.model = None
#         self.context = None
#         print("Setting up tokenizer")
#         model_name = "mistralai/Mistral-7B-Instruct-v0.2"
#         self.tokenizer= AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_fast = True)
#         self.tokenizer.pad_token = self.tokenizer.unk_token
#         self.tokenizer.pad_token_id =  self.tokenizer.unk_token_id
#         self.tokenizer.padding_side = 'left'


#         print("Setting up quantitation")
#         compute_dtype = getattr(torch, "float16")
#         bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_compute_dtype=compute_dtype,
#                 bnb_4bit_use_double_quant=True,
#         )

#         print("Loading model")
#         self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name, quantization_config=bnb_config,
#                 device_map="auto",
#         )

#     def encode(self, messages: list[dict]) -> torch.Tensor:
#         return self.tokenizer.encode(messages, return_tensors="pt").to("cuda")

#     def decode(self, output: torch.Tensor) -> str:
#         return self.tokenizer.decode(output)

#     def create_message_dict(self, is_user: bool, content: str) -> dict:
#         role = "user" if is_user else "assistant"
#         return {"role": role, "content": content}

#     def answer(self, prompt: str) -> str:
#         if self.context is None:
#             self.context = self.encode(self.tokenizer.eos_token + f"[INST] {prompt} [/INST]")
#         else:
#             self.context = torch.cat([self.context, self.encode(self.tokenizer.eos_token + f"[INST] {prompt} [/INST]")], dim=-1)
#         attention_mask = torch.ones(self.context.shape)
#         answer = None
#         for index in range(100):
#             self.tokenizer.padding_side = 'left'
#             output = self.model.generate(
#                 self.context, do_sample=True, max_length=self.context.shape[-1] + 1, 
#                 attention_mask=attention_mask, pad_token_id=self.tokenizer.eos_token_id)
#             last_token = output[..., -1:]
#             if answer is None:
#                 answer = last_token
#             else:
#                 answer = torch.cat([answer, last_token], dim=-1)
#             if index % 10 == 9:
#                 print(self.tokenizer.batch_decode(answer[0], skip_special_tokens=True))
#             self.context = torch.cat([self.context, last_token], dim = -1)
#             attention_mask = torch.cat([attention_mask, torch.ones((1, 1))], dim=-1)

        
#     def chat(self) -> None:
#         print(self.formatter.welcome())
#         for n in range(self.max_prompts):
#             prompt = input(self.formatter.prompt(n))
#             print(self.formatter.header(n))
#             self.answer(prompt)
#             print('\n' + self.formatter.footer(n))


def loadData(filePath: str) -> tokenization_utils_base.BatchEncoding:
    """
    Loads and tokenize data from filePath.
    File should be a pdf.
    """
    reader = PdfReader(filePath)
    sentences: list[str] = []
    relavent_pages = reader.pages[8:266]
    for index, page in enumerate(relavent_pages):
        text: str = page.extract_text()
        text = ''.join((char for char in text if char.isalnum() or char.isspace() or char in '.?!'))
        sentences.extend(re.split('[.?!\n]', text))
        logger.debug(f"Extracted text from page  {index}: \n{text}")
    logger.debug(f"The sentences list is: {sentences}")
    logger.info(f"Finished parsing {filePath}: {len(sentences)} sentences were extracted")
    return tokenizer(sentences, padding=True, truncation= True, return_tensors='pt')
    
    
def main():
   data = loadData('./DIP-TB.pdf')
   logger.info(data)
   trainer = Trainer(
       model=model,
       args=trainingArgs,
       train_dataset=data,6
       eval_dataset=data
                     )
   logger.info("Starting training")
   trainer.train()
    
if __name__ == '__main__' :
    # Create the parser and add argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='INFO', help='Set log level')

    # Parse the arguments
    args = parser.parse_args()

    # Set the logging level
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log}')
    logging.basicConfig(level=numeric_level)

    main()


