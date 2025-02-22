from datasets import load_dataset

class DatasetBuilder:
    def __init__(self, config):
        self.config = config

    def load_data(self, path):
        dataset = load_dataset("json", data_files=path)["train"]
        return dataset.map(self._format_entry, batched=True)

    def _format_entry(self, examples):
        return {
            "text": f"Question: {examples['question']}\nAnswer: {examples['answer']}"
        }
