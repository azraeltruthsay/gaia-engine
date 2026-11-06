"""Stock Gemma 4 format, matched against gemma-4-12B-it's own template (2qus)."""
from types import SimpleNamespace

from gaia_engine.core import ChatFormatter
from gaia_engine.prompt_format import normalize, stock_gemma4

CONVO = [{"role": "system", "content": "You are GAIA."}, {"role": "user", "content": "Hi there"},
         {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "How are you?"}]
# Rendered by transformers from the model's chat_template.jinja, <bos> removed.
REF_NO_THINK = ("<|turn>system\nYou are GAIA.<turn|>\n<|turn>user\nHi there<turn|>\n<|turn>model\nHello!<turn|>\n"
                "<|turn>user\nHow are you?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
REF_THINK = ("<|turn>system\n<|think|>\nYou are GAIA.<turn|>\n<|turn>user\nHi there<turn|>\n<|turn>model\nHello!<turn|>\n"
             "<|turn>user\nHow are you?<turn|>\n<|turn>model\n")


def test_matches_the_models_own_template():
    assert stock_gemma4(CONVO, enable_thinking=False) == REF_NO_THINK
    assert stock_gemma4(CONVO, enable_thinking=True) == REF_THINK
    assert stock_gemma4([{"role": "user", "content": "Hi"}], enable_thinking=True) == \
        "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"


def test_house_stays_the_default():
    assert normalize("") == "house" and normalize("Stock") == "stock" and normalize("odd") == "house"
    tok = SimpleNamespace(sot_token="<|turn>")
    house = ChatFormatter(tok)
    assert house.format_conversation([{"role": "user", "content": "Hi"}], enable_thinking=False).startswith(
        "<|turn>user<turn|>\nHi")
    stock = ChatFormatter(tok, prompt_format="stock")
    assert stock.format_conversation(CONVO, enable_thinking=False) == REF_NO_THINK
    assert stock.assistant_prefix(enable_thinking=False) == "<|turn>model\n<|channel>thought\n<channel|>"
