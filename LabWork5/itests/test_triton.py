import os
import requests

TRITON_HTTP = os.environ["TRITON_HTTP"].rstrip("/")
MODEL = os.environ["TRITON_MODEL"]
INP = os.environ["TRITON_INPUT"]
OUT = os.environ["TRITON_OUTPUT"]
EMB_DIM = int(os.environ["EMB_DIM"])


def test_triton_ready():
    r = requests.get(f"{TRITON_HTTP}/v2/health/ready", timeout=10)
    assert r.status_code == 200, r.text


def test_triton_infer_embedding_shape():
    url = f"{TRITON_HTTP}/v2/models/{MODEL}/infer"
    payload = {
        "inputs": [{
            "name": INP,
            "datatype": "BYTES",
            "shape": [1, 2],          # batch=1, items=2
            "data": ["hello", "world"]
        }],
        "outputs": [{"name": OUT}],
    }

    r = requests.post(url, json=payload, timeout=120)
    assert r.status_code == 200, r.text

    out = {o["name"]: o for o in r.json().get("outputs", [])}[OUT]

    shape = out.get("shape")
    data = out.get("data")

    assert isinstance(shape, list) and len(shape) >= 2, shape
    assert isinstance(data, list), type(data)

    # Triton отдаёт data плоским списком, shape описывает как его "собрать"
    assert shape[-2] == 2, f"expected 2 items, got shape={shape}"
    assert shape[-1] == EMB_DIM, f"expected dim={EMB_DIM}, got shape={shape}"
    assert len(data) == shape[-2] * shape[-1], f"len(data)={len(data)} shape={shape}"
