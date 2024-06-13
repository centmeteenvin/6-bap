class Settings:
    modelName: str = ""
    accesToken : str = ""
    trainingOutput: str = ""
    batchSize: int = 0
    epochs: int = 0
    evalBatchSizeMultiplier : int = 0
    gradientAccumulatorMultiplier : int = 0
    shouldTrain : bool = False
    modelPath: str | None = None
    shouldQuantize: bool = False
    autoMap: bool = False
    optimizer: str  = ""
    shouldChat: bool = False
    modelSavePath: str = ""