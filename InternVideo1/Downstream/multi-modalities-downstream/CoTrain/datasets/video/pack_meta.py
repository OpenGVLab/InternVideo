import pandas as pd
import numpy as np
from typing import Union

SEP = "<<<sep>>>"


class DummyMeta(object):
    def __init__(self, l):
        self._len = l

    def __len__(self):
        return self._len


def string_to_sequence(s: Union[str, list], dtype=np.int32) -> np.ndarray:
    if isinstance(s, list):
        assert not any(SEP in x for x in s)
        s = SEP.join(s)
    return np.array([ord(c) for c in s], dtype=dtype)


def sequence_to_string(seq: np.ndarray) -> Union[str, list]:
    s = ''.join([chr(c) for c in seq])
    if SEP in s:
        return s.split(SEP)
    return s


def pack_sequences(seqs: Union[np.ndarray, list]) -> (np.ndarray, np.ndarray):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets


def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]


def pack_metadata(obj: object, df: pd.DataFrame):
    assert not hasattr(obj, "metadata_keys")
    assert not hasattr(obj, "metadata_is_str")

    df = df.dropna()

    metadata_keys = list(df.columns)
    metadata_is_str = {c: df[c].dtype == pd.StringDtype for c in df.columns}
    for c in df.columns:
        if df[c].dtype == pd.StringDtype:
            assert not hasattr(obj, "metadata_{}_v".format(c))
            assert not hasattr(obj, "metadata_{}_o".format(c))

            seq_v, seq_o = pack_sequences([string_to_sequence(s) for s in df[c]])
            setattr(obj, "metadata_{}_v".format(c), seq_v)
            setattr(obj, "metadata_{}_o".format(c), seq_o)
        else:
            assert not hasattr(obj, "metadata_{}".format(c))

            seq = df[c].to_numpy()
            setattr(obj, "metadata_{}".format(c), seq)

    setattr(obj, "metadata_keys", metadata_keys)
    setattr(obj, "metadata_is_str", metadata_is_str)

    return DummyMeta(len(df))


def unpack_metadata(obj: object, i: int):
    ret = []
    for c in getattr(obj, "metadata_keys"):
        if not getattr(obj, "metadata_is_str")[c]:
            ret.append(getattr(obj, "metadata_{}".format(c))[i])
        else:
            ret.append(
                sequence_to_string(
                    unpack_sequence(
                        getattr(obj, "metadata_{}_v".format(c)),
                        getattr(obj, "metadata_{}_o".format(c)),
                        i,
                    )
                )
            )

    return pd.Series(dict(zip(obj.metadata_keys, ret)))
