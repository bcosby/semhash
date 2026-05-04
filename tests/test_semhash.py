from semhash import SemHashIndex


def test_add_and_search():
    index = SemHashIndex(bits=256, embedding_dim=128)
    index.add_texts(["refund policy", "hello world", "database backup"])
    out = index.search("how can i get a refund", k=2)
    assert len(out) == 2
    assert out[0].hamming_distance <= out[1].hamming_distance


def test_save_load_roundtrip(tmp_path):
    index = SemHashIndex(bits=256, embedding_dim=64)
    index.add_texts(["a", "b"])
    p = tmp_path / "idx.json"
    index.save(p)
    loaded = SemHashIndex.load(p)
    assert loaded.bits == 256
    assert loaded.search("a", 1)[0].text in {"a", "b"}
