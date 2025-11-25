from __future__ import annotations

from ._attrs import by_element


def get_attributes_for_ele(ele_keyword: str):
    try:
        return by_element[ele_keyword.upper()]
    except KeyError:
        pass

    for key in by_element:
        if key.startswith(ele_keyword.upper()):
            return by_element[key]
    raise KeyError(f"Element keyword not found: {ele_keyword.upper()}")


def get_attribute(ele_keyword: str, name: str):
    attrs = get_attributes_for_ele(ele_keyword)
    return attrs[name.upper()]
