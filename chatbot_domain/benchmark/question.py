from abc import ABC, abstractmethod
from enum import Enum
from json import load

class QuestionCategory(Enum):
    LITERAL = "literal",
    DEDUCTION = "deduction",
    OUTSIDE_DOMAIN = "outside domain",
    WRONG_ANSWERS = "wrong answers",
    
    def __eq__(self, value: object) -> bool:
        if type(value) == str:
            return self.value[0] == value
        return super().__eq__(value)


class Question(ABC):
    def __init__(self, question: str, options: list[str]) -> None:
        super().__init__()
        self.question = question
        self.options = options

    @abstractmethod
    def evaluate(self, answer: str) -> bool:
        """Returns true if given answer is correct"""
        
    def __repr__(self) -> str:
        return f"""
    {self.__class__.__name__}:
    questions: {self.question}
    options: {
        self.options
    }
    """


class LiteralQuestion(Question):
    """
    Question of which the answer can be found literally in the text.
    self.options[0] always contains the correct answer
    """

    def __init__(self, question: str, options: list[str]) -> None:
        super().__init__(question, options)

    def evaluate(self, answer: str) -> bool:
        if answer == self.options[0]:
            return True
        return False


class DeductionQuestion(Question):
    """
    Question for which you need to piece multiple pieces of information together to get to a correct answer.
    self.options[0] always contains the correct answer
    """
    def __init__(self, question: str, options: list[str]) -> None:
        super().__init__(question, options)
    
    def evaluate(self, answer: str) -> bool:
        if answer == self.options[0]:
            return True
        return False
    
class WrongQuestion(Question):
    """
    Question for which no correct options are provided.
    They should be answered with: "No correct answer is present."
    """
    answer = "No correct answer is present."
    
    def __init__(self, question: str, options: list[str]) -> None:
        super().__init__(question, options)
    
    def evaluate(self, answer: str) -> bool:
        return answer == WrongQuestion.answer
    

class OutsideDomainQuestion(Question):
    """
    Questions which are outside the domain.
    They should be answered with: "This question is outside my domain of knowledge." 
    """
    answer = "This question is outside my domain of knowledge." 
    
    def __init__(self, question: str, options: list[str]) -> None:
        super().__init__(question, options)
        
    def evaluate(self, answer: str) -> bool:
        return answer == OutsideDomainQuestion.answer
    
class QuestionFactory():
    def createQuestion(data: dict[
        "question": str,
        "options": list[str],
        "category": str,
        ]) -> Question:
        """Creates a Question object given the data"""
        question = data["question"]
        options = data["options"]
        category = data["category"]
        match category:
            case QuestionCategory.LITERAL:
                return LiteralQuestion(question, options)
            case QuestionCategory.DEDUCTION:
                return DeductionQuestion(question, options)
            case QuestionCategory.OUTSIDE_DOMAIN:
                return OutsideDomainQuestion(question, options)
            case QuestionCategory.WRONG_ANSWERS:
                return WrongQuestion(question, options)
            case _:
                raise Exception(f"Unknown category: {category}")
            
class QuestionParser(ABC):
    """
    Creates a list of questions from a certain source.
    The datasource is given in the constructor.
    """
    @abstractmethod
    def parse(self) -> list[Question]:
        pass
    
class JsonQuestionParser(QuestionParser):
    """
    Parses a json object which contains a list of dict question object in the 'questions' key.
    """
    def __init__(self, jsonFile: str) -> None:
        super().__init__()
        self.file = jsonFile
        
    def parse(self) -> list[Question]:
        with open(self.file, 'r') as file:
            data = load(file)
        questions: list[Question] = []
        for entry in data['questions']:
            questions.append(QuestionFactory.createQuestion(entry))
        return questions