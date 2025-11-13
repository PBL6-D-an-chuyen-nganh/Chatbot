import pytest
from intents import intent_utils
from utils import _extract_list_from_response


def test_extract_list_from_response_with_content():
    obj = {"content": [{"title": "A"}, {"title": "B"}], "page": 0}
    lst = _extract_list_from_response(obj)
    assert isinstance(lst, list) and len(lst) == 2


def test_extract_list_from_response_with_list():
    obj = [{"title": "A"}]
    lst = _extract_list_from_response(obj)
    assert lst == obj


def test_extract_list_from_response_with_plain_dict():
    obj = {"title": "Only"}
    lst = _extract_list_from_response(obj)
    assert isinstance(lst, list) and lst[0] == obj
