from tabs.pipeline import filters


def test_apply_rule_of_5_returns_none_outputs_for_empty_input():
    status, passed, df = filters.apply_rule_of_5(None)
    assert passed is None
    assert df is None
    assert "No compounds" in status


def test_apply_rule_of_5_marks_valid_compound_as_passed():
    compounds = [
        {
            "name": "Ethanol",
            "smiles": "CCO",
            "activity_value": 10,
            "activity_units": "nM",
        }
    ]

    status, passed, df = filters.apply_rule_of_5(compounds)

    assert passed is not None
    assert len(passed) == 1
    assert passed[0]["ro5_pass"] is True
    assert passed[0]["ro5_violations"] == 0
    assert df is not None
    assert "1 of 1" in status


def test_apply_adme_filter_returns_none_outputs_for_empty_input():
    status, passed, df = filters.apply_adme_filter(None)
    assert passed is None
    assert df is None
    assert "No compounds" in status


def test_apply_adme_filter_marks_valid_compound_as_passed():
    docked = [
        {
            "name": "Ibuprofen",
            "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "vina_affinity": -7.5,
            "binding_rank": 1,
            "mw": 206.3,
            "logp": 3.5,
            "hbd": 1,
            "hba": 1,
        }
    ]

    status, passed, df = filters.apply_adme_filter(docked)

    assert passed is not None
    assert len(passed) == 1
    assert passed[0]["adme_pass"] is True
    assert passed[0]["adme_score"] >= 1
    assert df is not None
    assert "1 of 1" in status