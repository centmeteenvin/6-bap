from random import shuffle
from copy import deepcopy
import datetime
from .question import Question, OutsideDomainQuestion, WrongQuestion, LiteralQuestion, DeductionQuestion
from shutil import get_terminal_size
from .testSubjects import TestSubject
from chatbot_domain import logger
import os

class TestScore:
    """This contains the results of an evaluation"""
    def __init__(self, subject = TestSubject) -> None:
        self.subject = subject
        self.results: list[dict["question": Question, "answer": str, "isCorrect": bool]] = []
    
    def addResult(self, question: Question, answer: str,isCorrect: bool) -> None:
        self.results.append({
            "question": question,
            "answer": answer,
            "isCorrect": isCorrect
        })
        
    @property
    def total(self) -> int:
        return sum([1 if result["isCorrect"] == True else 0 for result in self.results])
        
    @property
    def average(self) -> float:
        return self.total/len(self.results)
        
    def save(self, directory: str) -> None:
        """Save the testscore in the given directory under the name {self.subject.getname}-{score}-{datetime}"""
        score = self.total
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f"{directory}/{self.subject.getName}-{score} out {len(self.results)}-{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M')}.score", 'w') as file:
            file.write(self.__repr__() + '\n')
            for i, result in enumerate(self.results):
                question: Question = result["question"]
                file.write(f"{i+1}/{len(self.results)} {125*'='}\n")
                file.write("question: " + repr(question) +'\n')
                file.write("answer: " + repr(result["answer"]) + '\n')
                file.write("isCorrect: " + repr(result["isCorrect"]) + '\n')
            file.flush()
        
    def __repr__(self) -> str:
        overalScore = self.total
        literalScore = 0
        literalCount = 0
        deductionScore = 0
        deductionCount = 0
        wrongScore = 0
        wrongCount = 0
        oodScore = 0
        oodCount = 0
        for result in self.results:
            question = result["question"]
            isCorrect = 1 if result["isCorrect"] == True else 0
            if type(question) is LiteralQuestion:
                literalCount += 1
                literalScore += isCorrect
            elif type(question) is DeductionQuestion:
                deductionCount += 1
                deductionScore += isCorrect
            elif type(question) is WrongQuestion:
                wrongCount += 1
                wrongScore += isCorrect
            elif type(question) is OutsideDomainQuestion:
                oodCount += 1
                oodScore += isCorrect
                
        return f"""
    {get_terminal_size().columns * '='}
    # TEST RESULTS
    overal Score: {overalScore}/{len(self.results)}
    ## breakdown:
        literal:        {literalScore:10}/{literalCount}
        deduction:      {deductionScore:10}/{deductionCount}
        wrong answers:  {wrongScore:10}/{wrongCount}
        outside domain: {oodScore:10}/{oodCount}        
    {get_terminal_size().columns * '='}
    """
    


class Benchmarker():
    """
    This class takes a TestSubject and a list of questions.
    Run the evaluate method to evaluate the test subject and returns a TestScore
    """
    def __init__(self, testSubject: TestSubject, questions: list[Question]) -> None:
        self._testSubject = testSubject
        self._questions = questions
        
    def evaluate(self) -> TestScore:
        testScore = TestScore(self._testSubject)
        shuffledQuestions = deepcopy(self._questions)
        shuffle(shuffledQuestions)
        for i, question in enumerate(shuffledQuestions):
            logger.info(f"Question {i+1}/{len(self._questions)}")
            #Shuffle the initial options
            options = deepcopy(question.options)
            shuffle(options)
            
            #Append 2 standard options
            options.append(str(WrongQuestion.answer))
            options.append(str(OutsideDomainQuestion.answer))
            
            #Prepend numbers before each option
            numberedOptions = []
            for i, option in enumerate(options):
                numberedOptions.append(f"{i + 1}. {option}")
            
            #Ask the question to the testSubject
            # selectedOption = self._testSubject.askQuestion(question.question, numberedOptions)
            # answer = options[selectedOption]
            answer = options[1]
            
            #Check if the answer is correct
            isCorrect = question.evaluate(answer)
            
            #Add result to the testScore
            testScore.addResult(question, answer, isCorrect)
        return testScore
