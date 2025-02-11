from langchain.prompts import PromptTemplate

class BasicPromptTemplate:
    """A basic prompt template class for simple questions with context."""
    template = """
    Given the following context, answer the question accurately and concisely.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    @classmethod
    def get_template(cls):
        return PromptTemplate(template=cls.template, input_variables=["question", "context"])


class DetailedExplanationPromptTemplate:
    """A template that asks for detailed explanations and reasoning."""
    template = """
    Use the provided context to give a detailed, step-by-step answer to the question below.

    Context:
    {context}

    Question:
    {question}

    Detailed Answer:
    """

    @classmethod
    def get_template(cls):
        return PromptTemplate(template=cls.template, input_variables=["question", "context"])
