
from scripts.benchmark import download_all_datasets as module


def test_download_all_records_downloads_and_manual_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "DATASET_SPECS", {"automatic": object(), "manual": object()})

    def fake_download(dataset_id, *, cache_dir, limit, manual_file):
        if dataset_id == "manual":
            raise NotImplementedError("supply export")
        path = cache_dir / dataset_id / "test.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('{"id": 1}\n', encoding="utf-8")
        return path

    monkeypatch.setattr(module, "download_external_dataset", fake_download)
    summary = module.download_all(tmp_path, limit=10, manual_dir=None)

    assert summary["downloaded"] == 1
    assert summary["manual_action_required"] == 1
    assert summary["failed"] == 0
    assert (tmp_path / "download_manifest.json").exists()
