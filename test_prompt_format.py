"""Unit tests for L prompt formatting (no NDIF needed)."""
from ndif import _dialogue_to_messages, _format_l_for_qwen, channel_prompt


class FakeTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=None, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{m['content']}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)


def test_strips_assistant_suffix_and_parses_turns():
    text = "user: hi\nassistant: hello\nuser: what is 2+2\nassistant:"
    msgs = _dialogue_to_messages(text)
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what is 2+2"},
    ]


def test_literal_backslash_n_suffix_also_stripped():
    # Regression: the old bug looked for the characters \\n, not a newline.
    text = "user: hi\\nassistant:"
    msgs = _dialogue_to_messages(text)
    assert msgs[-1]["role"] == "user"
    assert "assistant" not in msgs[-1]["content"]


def test_qwen_format_uses_chat_template():
    out = _format_l_for_qwen(FakeTok(), "user: Name the capital of France.\nassistant:")
    assert "<|assistant|>" in out
    assert "capital of France" in out
    assert out.strip().endswith("<|assistant|>")


def test_channel_band_outside_speech():
    rendered = "<|user|>\nHi\n<|assistant|>\n"
    banded = channel_prompt(rendered, ["R", "B"], {}, default=4)
    assert banded.startswith(" ?")
    assert banded.endswith(rendered)
