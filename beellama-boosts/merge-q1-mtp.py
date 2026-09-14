#!/usr/bin/env python3
"""Merge Q1_0 trunk with vinpix MTP block into Q1+MTP GGUF.

Usage:
    python3 merge-q1-mtp.py \
        /mnt/nvmedata/models/bonsai-27b/Bonsai-27B-Q1_0.gguf \
        /mnt/nvmedata/models/ternary-bonsai-27b/Ternary-Bonsai-27B-MTP-Q2_K.gguf \
        /mnt/nvmedata/models/bonsai-27b/Bonsai-27B-Q1_0-MTP.gguf
"""
import sys
sys.path.insert(0, "gguf-py")
import gguf
from gguf.constants import GGMLQuantizationType

Q1_PATH, MTP_PATH, OUT_PATH = sys.argv[1], sys.argv[2], sys.argv[3]

q1 = gguf.GGUFReader(Q1_PATH)
mt = gguf.GGUFReader(MTP_PATH)

VT = gguf.GGUFValueType


def copy_kv(writer, reader, skip=()):
    for f in reader.fields.values():
        if f.name in skip or f.name.startswith("GGUF."):
            continue
        v = f.contents()
        t = f.types[0] if isinstance(f.types, list) else f.types
        if t == VT.UINT8:
            writer.add_uint8(f.name, v)
        elif t == VT.INT8:
            writer.add_int8(f.name, v)
        elif t == VT.UINT16:
            writer.add_uint16(f.name, v)
        elif t == VT.INT16:
            writer.add_int16(f.name, v)
        elif t == VT.UINT32:
            writer.add_uint32(f.name, v)
        elif t == VT.INT32:
            writer.add_int32(f.name, v)
        elif t == VT.FLOAT32:
            writer.add_float32(f.name, v)
        elif t == VT.BOOL:
            writer.add_bool(f.name, bool(v))
        elif t == VT.STRING:
            writer.add_string(f.name, v)
        elif t == VT.ARRAY:
            writer.add_array(f.name, list(v))
        elif t == VT.UINT64:
            writer.add_uint64(f.name, v)
        elif t == VT.INT64:
            writer.add_int64(f.name, v)
        elif t == VT.FLOAT64:
            writer.add_float64(f.name, v)
        else:
            raise ValueError(f"unhandled kv type {t} for {f.name}")


def add_raw(writer, tensor):
    import numpy as np
    raw = np.asarray(tensor.data)
    if raw.dtype != np.uint8:
        raw = raw.view(np.uint8)
    assert raw.nbytes == tensor.n_bytes, (tensor.name, raw.nbytes, tensor.n_bytes)
    writer.add_tensor(tensor.name, raw, raw_shape=tuple(int(d) for d in raw.shape),
                      raw_dtype=tensor.tensor_type)


mtp_tensors = [t for t in mt.tensors if t.name.startswith("blk.64.")]
assert len(mtp_tensors) == 15, len(mtp_tensors)
mtp_names = set(t.name for t in mtp_tensors)
assert not any(t.name in mtp_names for t in q1.tensors), "name collision"

n_nextn = mt.get_field("qwen35.nextn_predict_layers").contents()
print(f"nextn_predict_layers = {n_nextn}")

w = gguf.GGUFWriter(OUT_PATH, "qwen35")
copy_kv(w, q1, skip=("general.name", "qwen35.block_count"))
w.add_string("general.name", "Bonsai-27B-Q1_0-MTP")
import re as _re
trunk_idx = sorted(set(int(t.name.split(".")[1]) for t in q1.tensors
                       if t.name.startswith("blk.") and t.name.split(".")[1].isdigit()))
n_trunk_blocks = len(trunk_idx)
w.add_uint32("qwen35.block_count", n_trunk_blocks + int(n_nextn))
w.add_uint32("qwen35.nextn_predict_layers", int(n_nextn))
print(f"block_count = {n_trunk_blocks} + {int(n_nextn)} MTP")

for t in q1.tensors:
    add_raw(w, t)
print(f"trunk tensors: {len(q1.tensors)}")
for t in mtp_tensors:
    add_raw(w, t)
print(f"mtp tensors: {len(mtp_tensors)}")

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"WROTE {OUT_PATH}")
