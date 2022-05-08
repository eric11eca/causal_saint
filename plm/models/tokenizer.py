from plm.shared.model_resolution import ModelArchitectures, TOKENIZER_CLASS_DICT


class PLM_Tokenizer:

    def __init__(self, model_name) -> None:
        self.tokenizer = None
        self.initialize_tokenizer(model_name)

    def initialize_tokenizer(self, model_name):
        if not model_name in ModelArchitectures:
            raise Exception(
                "This model is not supported by the current framework!")
        self. tokenizer = TOKENIZER_CLASS_DICT[model_name]

    def tokenize_example(self, examples, batched=False):
        if not batched:
            tokenized_examples = []
            for example in examples:
                tokenized = self.tokenizer.tokenize(
                    example["premise"], example["hypothesis"])
                tokenized_examples.append(tokenized)
            return tokenized_examples
