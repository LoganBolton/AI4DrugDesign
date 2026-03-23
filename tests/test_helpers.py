from tabs.pipeline import helpers


class DummyResponse:
    def __init__(self, ok=True, json_data=None):
        self.ok = ok
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


def test_get_openai_client_returns_none_when_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = helpers.get_openai_client()
    assert client is None


def test_get_uniprot_from_pdb_returns_first_accession(monkeypatch):
    def fake_get(url, timeout=15):
        return DummyResponse(
            ok=True,
            json_data={
                "1abc": {
                    "UniProt": {
                        "P12345": {},
                        "Q99999": {}
                    }
                }
            },
        )

    monkeypatch.setattr(helpers.requests, "get", fake_get)
    result = helpers.get_uniprot_from_pdb("1ABC")
    assert result == "P12345"


def test_get_uniprot_from_pdb_returns_none_when_no_mapping(monkeypatch):
    def fake_get(url, timeout=15):
        return DummyResponse(ok=True, json_data={"1abc": {"UniProt": {}}})

    monkeypatch.setattr(helpers.requests, "get", fake_get)
    result = helpers.get_uniprot_from_pdb("1ABC")
    assert result is None


def test_fetch_pdb_ligands_filters_skipped_ligands(monkeypatch):
    entry = {
        "rcsb_entry_container_identifiers": {
            "non_polymer_entity_ids": ["1", "2"]
        }
    }

    def fake_get(url, timeout=10):
        if url.endswith("/1ABC/1"):
            return DummyResponse(
                ok=True,
                json_data={"pdbx_entity_nonpoly": {"comp_id": "ATP", "name": "ATP"}}
            )
        if url.endswith("/1ABC/2"):
            return DummyResponse(
                ok=True,
                json_data={"pdbx_entity_nonpoly": {"comp_id": "HOH", "name": "Water"}}
            )
        return DummyResponse(ok=False, json_data={})

    monkeypatch.setattr(helpers.requests, "get", fake_get)
    ligands = helpers.fetch_pdb_ligands("1ABC", entry=entry)

    assert ligands == [{"comp_id": "ATP", "name": "ATP"}]


def test_fetch_pdb_ligands_returns_empty_list_on_entry_request_failure(monkeypatch):
    def fake_get(url, timeout=15):
        raise Exception("network error")

    monkeypatch.setattr(helpers.requests, "get", fake_get)
    ligands = helpers.fetch_pdb_ligands("1ABC")

    assert ligands == []