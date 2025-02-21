class DatasetBuilder:
    """
    Processes raw JSON data into training-ready format
    Implements:
    - Dynamic padding
    - Chain-of-thought formatting
    - Dataset versioning
    """
    def format_entries(self, examples):
        return {
            "text": f"Question: {examples['question']}\nReasoning: {examples['cot']}\nAnswer: {examples['answer']}"
        }
    
    def normalize_text(self, text):
        return text.strip().replace("\n", "\\n")
