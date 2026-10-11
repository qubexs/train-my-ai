"""backends.py: unified transcript policy. No binaries needed."""
from backends import transcript_messages, transcript_text


def test_transcript_caps_and_labels():
    msgs = [{"role": "user", "content": "u" * 1500},
            {"role": "assistant", "content": "a" * 800},
            {"role": "tool", "name": "run", "content": "r"}]
    tt = transcript_text(msgs)
    tm = transcript_messages(msgs)
    assert len(tt) == len(tm) == 3
    # content caps (prefix labels excluded)
    assert tt[0].endswith("u" * 1000) and tt[1].endswith("a" * 500)
    assert "[Hasil tool run]:" in tt[2]
    assert tm[2] == {"role": "user", "content": "[Hasil tool run]: r"}
    assert tm[0]["content"].endswith("u" * 1000)


def test_transcript_recent_only():
    msgs = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    assert len(transcript_text(msgs)) == 6
    assert transcript_text(msgs)[0].endswith("q14")
