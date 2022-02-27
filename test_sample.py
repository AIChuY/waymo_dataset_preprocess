""" content of test_sample.py
"""


def func(x_value):
    """ "Test function"""
    return x_value + 1


def test_answer():
    """ "Test placeholder"""
    assert func(4) == 5
