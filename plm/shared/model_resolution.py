import transformers

from enum import Enum
from dataclasses import dataclass
from plm.utils.datastructures import BiDict


class ModelArchitectures(Enum):
    BERT = "bert"
    ROBERTA = "roberta"
    ALBERT = "albert"
    ELECTRA = "electra"
    DEBERTA = "deberta"
    DEBERTAV2 = "deberta-v2"
    DEBERTAV3 = "deberta-v3"
    XLM_ROBERTA = "xlm-roberta"
    XLM = "xlm"
    BART = "bart"
    MBART = "mbart"
    T5 = "t5"
    GPT2 = "gpt-2"

    @classmethod
    def from_model_type(cls, model_type: str):
        return cls(model_type)


TOKENIZER_CLASS_DICT = BiDict(
    {
        ModelArchitectures.BERT: transformers.BertTokenizer,
        ModelArchitectures.XLM: transformers.XLMTokenizer,
        ModelArchitectures.ROBERTA: transformers.RobertaTokenizer,
        ModelArchitectures.XLM_ROBERTA: transformers.XLMRobertaTokenizer,
        ModelArchitectures.ALBERT: transformers.AlbertTokenizer,
        ModelArchitectures.BART: transformers.BartTokenizer,
        ModelArchitectures.MBART: transformers.MBartTokenizer,
        ModelArchitectures.ELECTRA: transformers.ElectraTokenizer,
        ModelArchitectures.DEBERTAV2: transformers.DebertaV2Tokenizer,
        ModelArchitectures.DEBERTAV3: transformers.DebertaV2Tokenizer,
        ModelArchitectures.DEBERTA: transformers.DebertaTokenizer,
        ModelArchitectures.T5: transformers.T5Tokenizer,
        ModelArchitectures.GPT2: transformers.GPT2Tokenizer
    }
)


@dataclass
class ModelClassSpec:
    config_class: type
    tokenizer_class: type
    model_class: type


def resolve_tokenizer_class(model_type):
    """Get tokenizer class for a given model architecture.

    Args:
        model_type (str): model shortcut name.

    Returns:
        Tokenizer associated with the given model.

    """
    return TOKENIZER_CLASS_DICT[ModelArchitectures(model_type)]


def resolve_model_arch_tokenizer(tokenizer):
    """Get the model architecture for a given tokenizer.

    Args:
        tokenizer

    Returns:
        ModelArchitecture

    """
    assert len(TOKENIZER_CLASS_DICT.inverse[tokenizer.__class__]) == 1
    return TOKENIZER_CLASS_DICT.inverse[tokenizer.__class__][0]


def resolve_is_lower_case(tokenizer):
    if isinstance(tokenizer, transformers.BertTokenizer):
        return tokenizer.basic_tokenizer.do_lower_case
    if isinstance(tokenizer, transformers.AlbertTokenizer):
        return tokenizer.do_lower_case
    else:
        return False


def bart_or_mbart_model_heuristic(model_config: transformers.BartConfig) -> ModelArchitectures:
    if model_config.is_valid_mbart():
        return ModelArchitectures.MBART
    else:
        return ModelArchitectures.BART
