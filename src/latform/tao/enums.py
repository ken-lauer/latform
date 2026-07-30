from __future__ import annotations

from collections.abc import Sequence

# Copied in manually from:
# * css4_coor_reference.py
# * qp_css4_colors_mod.f90 in Bmad
# Left in original index comments to validate copy/paste work

QP_ENUMS: dict[str, dict[int, str]] = {
    "color": {
        0: "white",
        1: "black",
        2: "red",
        3: "green",
        4: "blue",
        5: "cyan",
        6: "magenta",
        7: "yellow",
        8: "orange",
        9: "yellow_green",
        10: "light_green",
        11: "navy_blue",
        12: "purple",
        13: "reddish_purple",
        14: "dark_grey",
        15: "light_grey",
        16: "transparent",
        # CSS4 colors
        17: "lime",
        18: "maroon",
        19: "navy",
        20: "darkcyan",
        21: "indianred",  # index 21
        22: "slateblue",
        23: "olive",
        24: "violet",
        25: "aquamarine",
        26: "khaki",  # index 26
        27: "limegreen",
        28: "darkviolet",
        29: "sandybrown",
        30: "cadetblue",
        31: "yellowgreen",  # index 31
        32: "mediumorchid",
        33: "pink",
        34: "crimson",
        35: "lightgreen",
        36: "mediumturquoise",  # index 36
        37: "orangered",
        38: "cornflowerblue",
        39: "darkturquoise",
        40: "seagreen",
        41: "sienna",  # index 41
        42: "hotpink",
        43: "lightblue",
        44: "rebeccapurple",
        45: "mediumvioletred",
        46: "darkkhaki",  # index 46
        47: "lightseagreen",
        48: "darkgoldenrod",
        49: "gray",
        50: "tomato",
        51: "greenyellow",  # index 51
        52: "indigo",
        53: "firebrick",
        54: "lemonchiffon",
        55: "peru",
        56: "mediumblue",  # index 56
        57: "mediumseagreen",
        58: "palevioletred",
        59: "forestgreen",
        60: "darkorchid",
        61: "deepskyblue",  # index 61
        62: "mediumpurple",
        63: "royalblue",
        64: "gainsboro",
        65: "goldenrod",
        66: "mediumaquamarine",  # index 66
        67: "lightskyblue",
        68: "darkolivegreen",
        69: "chocolate",
        70: "burlywood",
        71: "olivedrab",  # index 71
        72: "steelblue",
        73: "darkseagreen",
        74: "rosybrown",
        75: "salmon",
        76: "gold",  # index 76
        77: "wheat",
        78: "darkslategray",
        79: "midnightblue",
        80: "plum",
        81: "silver",  # index 81
        82: "orchid",
        83: "saddlebrown",
        84: "dimgray",
        85: "darkslateblue",
        86: "dodgerblue",  # index 86
        87: "lavender",
        88: "mediumslateblue",
        89: "mistyrose",
        90: "thistle",
        91: "coral",  # index 91
        92: "darksalmon",
        93: "deeppink",
        94: "darkgreen",
        95: "lightslategray",
        96: "blueviolet",  # index 96
        97: "mediumspringgreen",
        98: "lightcyan",
        99: "darkorange",
        100: "lightsalmon",
        101: "paleturquoise",  # index 101
        102: "lightsteelblue",
        103: "honeydew",
        104: "bisque",
        105: "lightyellow",
        106: "lavenderblush",  # index 106
        107: "turquoise",
        108: "brown",
        109: "palegreen",
        110: "lightcoral",
        111: "aliceblue",  # index 111
        112: "lightgray",
        113: "darkmagenta",
        114: "teal",
        115: "palegoldenrod",
        116: "ivory",  # index 116
        117: "skyblue",
        118: "beige",
        119: "lightpink",
        120: "slategray",
        121: "papayawhip",  # index 121
        122: "tan",
        123: "peachpuff",
        124: "linen",
        125: "navajowhite",
        126: "whitesmoke",  # index 126
        127: "darkblue",
        128: "darkred",
        129: "moccasin",
        130: "mintcream",
        131: "blanchedalmond",  # index 131
        132: "seashell",
        133: "powderblue",
        134: "cornsilk",
        135: "ghostwhite",
        136: "lightgoldenrodyellow",  # index 136
        137: "snow",
        138: "azure",
        139: "antiquewhite",
        140: "oldlace",
        141: "floralwhite",  # index 141
        142: "lawngreen",
        143: "darkgray",
        144: "aqua",
        145: "chartreuse",
        146: "darkgrey",  # index 146
        147: "darkslategrey",
        148: "dimgrey",
        149: "fuchsia",
        150: "grey",
        151: "lightgrey",  # index 151
        152: "lightslategrey",
        153: "slategrey",
        154: "springgreen",
    },
    "line_pattern": {
        1: "solid",
        2: "dashed",
        3: "dash_dot",
        4: "dotted",
        5: "dash_dot3",
    },
    "fill_pattern": {
        1: "solid_fill",
        2: "no_fill",
        3: "hatched",
        4: "cross_hatched",
    },
    "symbol_type": {
        -1: "do_not_draw",
        0: "square",
        1: "dot",
        2: "plus",
        3: "times",
        4: "circle",
        5: "x_symbol",
        7: "triangle",
        8: "circle_plus",
        9: "circle_dot",
        10: "square_concave",
        11: "diamond",
        12: "star5",
        13: "triangle_filled",
        14: "red_cross",
        15: "star_of_david",
        16: "square_filled",
        17: "circle_filled",
        18: "star5_filled",
    },
    "arrow_head_type": {
        1: "filled_arrow_head",
        2: "outline_arrow_head",
    },
}

_FIELD_ENUM_KEYS: dict[tuple[str, str], str] = {
    ("line", "color"): "color",
    ("line", "pattern"): "line_pattern",
    ("symbol", "type"): "symbol_type",
    ("symbol", "color"): "color",
    ("symbol", "fill_pattern"): "fill_pattern",
}


def integer_enum_for_field(components: Sequence[str]) -> dict[int, str] | None:
    """
    The enum governing a character field, from its ``%``-path component names.

    ``components`` are the path names with array indices stripped, outermost
    first (e.g. ``("curve", "symbol", "type")``).

    Any other ``*color*`` leaf maps to the color enum. Returns ``None`` when no
    enum governs the field.
    """
    if not components:
        return None
    leaf = components[-1].lower()
    parent = components[-2].lower() if len(components) >= 2 else ""
    key = _FIELD_ENUM_KEYS.get((parent, leaf))
    if key is None and "color" in leaf:
        key = "color"
    return QP_ENUMS[key] if key is not None else None
