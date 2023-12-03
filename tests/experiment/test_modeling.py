from experiment import modeling


def test_pretty_fun():
    assert len(modeling.run()) == 5
