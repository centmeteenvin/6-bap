from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Pipeline
from ..settings import Settings
from .. import logger, torch
from peft import PeftModelForCausalLM, PeftConfig
from chatbot_domain.secrets import HUGGINGFACE_ACCESS

class Model:
    def __init__(self, modelName: str, shouldQuantize: bool, deviceMap: str) -> None:
        logger.info(f"loading model")
        bnb_config = None
        if shouldQuantize:
            logger.info("With Quantization")
            compute_dtype = getattr(torch, "float16")
            bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            )
        
        self.model : AutoModelForCausalLM=  AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=modelName,
            token = HUGGINGFACE_ACCESS,
            device_map = deviceMap,
            quantization_config = bnb_config 
        )
        self.name= modelName
        
    def delete(self):
        print("Deleted Model")
        del self.model
        torch.cuda.empty_cache()

class AdaptedModel(Model):
    def __init__(self,name: str, shouldQuantize: bool, deviceMap: str) -> None:
        self.config = PeftConfig.from_pretrained(name, token=HUGGINGFACE_ACCESS)
        super().__init__(self.config.base_model_name_or_path, shouldQuantize, deviceMap)
        self.model = PeftModelForCausalLM.from_pretrained(self.model, name, token=HUGGINGFACE_ACCESS)
        self.name = name