from transformers import Trainer, TrainingArguments
from ..settings import Settings
from ..transformer.model import Model
from datasets import DatasetDict
from .. import logger
from torch.cuda import empty_cache

class Training:
    def __init__(self, model: Model, dataSet: DatasetDict) -> None:
        logger.info(f"creating trainer")
        self._trainingArgs = self._argsCreator()
        self.model = model
        self.dataSet = dataSet
        self.trainer = self._trainerCreator()
        
    def _argsCreator(self) ->  TrainingArguments:
        return TrainingArguments(
            output_dir=Settings.trainingOutput,
            evaluation_strategy='epoch',
            per_device_train_batch_size=Settings.batchSize,
            per_device_eval_batch_size=Settings.batchSize * Settings.evalBatchSizeMultiplier,
            gradient_accumulation_steps=Settings.batchSize * Settings.gradientAccumulatorMultiplier,
            logging_dir='./logs',
            logging_steps= 5,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_steps=1,
            do_eval=True,
            do_train=True,
            optim=Settings.optimizer,
            num_train_epochs=Settings.epochs,
            save_strategy='epoch',
            save_steps=1,
        )
        
    def _trainerCreator(self) -> Trainer:
        return Trainer(
            model=self.model.model,
            args=self._trainingArgs,
            train_dataset=self.dataSet["train"],
            eval_dataset=self.dataSet["evaluation"],
        )
    
    def _restartTraining(self):
        del self._trainingArgs
        del self.trainer
        empty_cache()
        self._trainingArgs = self._argsCreator()
        self.trainer = self._trainerCreator()
        self.train()
        
        
    def train(self) -> None:
        try:    
            logger.info("Starting training")
            self.trainer.train()
            logger.info("Finished training")
            logger.info(f"Saving model to {Settings.modelSavePath}")
            self.trainer.save_model(Settings.modelSavePath)
        except Exception as e:
            logger.debug(f'encounter {e} while training')
            if Settings.batchSize != 1:
                Settings.batchSize = int(Settings.batchSize/2)
                logger.error(f"Problem during training, reducing batch size to {Settings.batchSize}")
                self._restartTraining()
                return
            if Settings.evalBatchSizeMultiplier != 1:
                Settings.evalBatchSizeMultiplier = int(Settings.evalBatchSizeMultiplier/2)
                logger.error(f"Problem during training, reducing eval batch size multiplier to {Settings.evalBatchSizeMultiplier} ")
                self._restartTraining()
                return
            if Settings.optimizer == "adamw_torch_fused":
                Settings.optimizer = "adamw_bnb_8bit"
                logger.error(f"Problem during training, changing optimizer to {Settings.optimizer}")
                self._restartTraining()
                return
            
            logger.fatal(f"Training could not be completed:\n{e}")
            del self.trainer
            del self._trainingArgs
            empty_cache()
            exit(420)            
                
            
