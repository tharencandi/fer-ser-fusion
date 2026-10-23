"""Hidden-state modules of the matched protocol.

Run from the repository root:  python -m pytest tests -q
Needs torch, transformers and funasr (no downloads: random-weight backbones).
"""
import pytest
import torch

from cv import models as M
from cv import testing as T


def test_full_size_emotion2vec_has_13_states():
    st = M.Emotion2VecStates(T.tiny_emotion2vec(full_size=True))
    assert st.num_states == 13 and st.hidden_size == 768
    with torch.no_grad():
        out = st(torch.randn(1, 16000))
    assert out.shape == (13, 49, 768)


@pytest.mark.parametrize("make", [lambda: M.Wav2Vec2States(M.TINY),
                                  lambda: M.Emotion2VecStates(T.tiny_emotion2vec())])
def test_states_stay_in_eval_mode_and_are_deterministic(make):
    st = make()
    st.train()
    assert not any(m.training for m in st.modules())
    x = torch.randn(1, 16000)
    with torch.no_grad():
        a, b = st(x), st(x)
    assert a.shape[0] == st.num_states
    assert torch.equal(a, b)
