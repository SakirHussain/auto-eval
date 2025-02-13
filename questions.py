class BasicQuestions:
    """A class representing a list of basic questions."""
    questions = [
        "Elucidate the crucial role of embodiment in the cognitive development process with an example of a warehouse robot that is designed to navigate through the cluttered environment. Give the details of the three types of embodiment hypothesis for the above scenario."
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
