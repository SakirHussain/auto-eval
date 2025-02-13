class FewShotExamples:
    """Class containing few-shot examples for problem reconstruction, condition extraction, and comparison."""

    @staticmethod
    def problem_reconstruction_examples():
        """Returns few-shot examples for problem reconstruction."""
        return """
        **Example 1:**
        Answer: "There are 10 groups because 90 divided by 9 equals 10."
        Problem: "There are 90 items that need to be divided into equal groups of 9. How many groups can be formed?"

        **Example 2:**
        Answer: "The total cost is $500 because 5 products each cost $100."
        Problem: "If each product costs $100, what is the total cost for 5 products?"

        **Example 3:**
        Answer: "The final speed is 20 m/s because the car accelerated by 5 m/s² for 4 seconds."
        Problem: "A car accelerates at a rate of 5 m/s² for 4 seconds. What is its final speed?"
        """

    @staticmethod
    def condition_extraction_examples():
        """Returns few-shot examples for condition extraction."""
        return """
        **Example 1:**
        Problem: "There are 90 items that need to be divided into equal groups of 9. How many groups can be formed?"
        Extracted Conditions:
        - "There are 90 items."
        - "Each group contains 9 items."

        **Example 2:**
        Problem: "A box contains 100 apples, and each basket can hold 10 apples."
        Extracted Conditions:
        - "A box contains 100 apples."
        - "Each basket holds 10 apples."

        **Example 3:**
        Problem: "A tank holds 50 liters of water, and each bucket carries 5 liters. How many buckets are needed?"
        Extracted Conditions:
        - "A tank holds 50 liters of water."
        - "Each bucket carries 5 liters."
        """

    @staticmethod
    def condition_comparison_examples():
        """Returns few-shot examples for condition comparison."""
        return """
        **Example 1:**
        Candidate Condition: "There are 90 people."
        Given Conditions:
        - "There are 90 items."
        - "Each group contains 9 items."
        Deduction: False
        Explanation: "The original problem refers to items, not people."

        **Example 2:**
        Candidate Condition: "Each basket can carry 10 items."
        Given Conditions:
        - "Each basket holds 10 apples."
        Deduction: True
        Explanation: "Apples are a type of item, so this deduction is valid."
        """
