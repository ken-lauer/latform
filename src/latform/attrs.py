from __future__ import annotations

from ._attrs import by_element as element_key_to_attrs


def get_attributes_for_ele(ele_keyword: str):
    try:
        return element_key_to_attrs[ele_keyword.upper()]
    except KeyError:
        pass

    for key in element_key_to_attrs:
        if key.startswith(ele_keyword.upper()):
            return element_key_to_attrs[key]
    raise KeyError(f"Element keyword not found: {ele_keyword.upper()}")


def get_attribute(ele_keyword: str, name: str):
    attrs = get_attributes_for_ele(ele_keyword)
    return attrs[name.upper()]
