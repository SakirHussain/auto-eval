class BasicQuestions:
    """A class representing a list of basic questions."""
    questions = [
        "What is the capital of France?",
        "What is Parthivi's favorite flower?",
    ]

    @classmethod
    def get_questions(cls):
        return cls.questions


class AdvancedQuestions:
    """A class for more advanced and detailed questions."""
    questions = [
        "Explain the role of chlorophyll in photosynthesis in detail.",
        "Describe the architectural components of a RAG-based system.",
        "How does retrieval improve the performance of language models?"
    ]

    @classmethod
    def get_questions(cls):
        return cls.questions
