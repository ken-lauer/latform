from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IntrinsicFunction:
    name: str
    arguments: list[str]
    min_args: int
    max_args: int
    description: str


INTRINSIC_FUNCTIONS = [
    IntrinsicFunction("sqrt", ["x"], 1, 1, "Square Root"),
    IntrinsicFunction("log", ["x"], 1, 1, "Logarithm"),
    IntrinsicFunction("exp", ["x"], 1, 1, "Exponential"),
    IntrinsicFunction("sin", ["x"], 1, 1, "Sine"),
    IntrinsicFunction("cos", ["x"], 1, 1, "Cosine"),
    IntrinsicFunction("tan", ["x"], 1, 1, "Tangent"),
    IntrinsicFunction("cot", ["x"], 1, 1, "Cotangent"),
    IntrinsicFunction("sinc", ["x"], 1, 1, "Sin(x)/x Function"),
    IntrinsicFunction("asin", ["x"], 1, 1, "Arc sine"),
    IntrinsicFunction("acos", ["x"], 1, 1, "Arc cosine"),
    IntrinsicFunction("atan", ["x"], 1, 1, "Arc tangent"),
    IntrinsicFunction("atan2", ["y", "x"], 2, 2, "Arc tangent of y/x"),
    IntrinsicFunction("sinh", ["x"], 1, 1, "Hyperbolic sine"),
    IntrinsicFunction("cosh", ["x"], 1, 1, "Hyperbolic cosine"),
    IntrinsicFunction("tanh", ["x"], 1, 1, "Hyperbolic tangent"),
    IntrinsicFunction("coth", ["x"], 1, 1, "Hyperbolic cotangent"),
    IntrinsicFunction("asinh", ["x"], 1, 1, "Hyperbolic arc sine"),
    IntrinsicFunction("acosh", ["x"], 1, 1, "Hyperbolic arc cosine"),
    IntrinsicFunction("atanh", ["x"], 1, 1, "Hyperbolic arc tangent"),
    IntrinsicFunction("acoth", ["x"], 1, 1, "Hyperbolic arc cotangent"),
    IntrinsicFunction("abs", ["x"], 1, 1, "Absolute Value"),
    IntrinsicFunction("factorial", ["n"], 1, 1, "Factorial"),
    IntrinsicFunction("ran", [], 0, 0, "Random number between 0 and 1"),
    IntrinsicFunction("ran_gauss", ["sig_cut"], 0, 1, "Gaussian distributed random number"),
    IntrinsicFunction("int", ["x"], 1, 1, "Nearest integer with magnitude less than x"),
    IntrinsicFunction("nint", ["x"], 1, 1, "Nearest integer to x"),
    IntrinsicFunction("sign", ["x"], 1, 1, "1 if x positive, -1 if negative, 0 if zero"),
    IntrinsicFunction("floor", ["x"], 1, 1, "Nearest integer less than x"),
    IntrinsicFunction("ceiling", ["x"], 1, 1, "Nearest integer greater than x"),
    IntrinsicFunction("modulo", ["a", "p"], 2, 2, "a - floor(a/p) * p. Will be in range [0, p]"),
    IntrinsicFunction("mass_of", ["A"], 1, 1, "Mass of particle A"),
    IntrinsicFunction(
        "charge_of",
        ["A"],
        1,
        1,
        "Charge, in units of the elementary charge, of particle A",
    ),
    IntrinsicFunction(
        "anomalous_moment_of", ["A"], 1, 1, "Anomalous magnetic moment of particle A"
    ),
    IntrinsicFunction("species", ["A"], 1, 1, "Species ID of A"),
]

RESERVED_NAMES = {
    "beam",
    "beginning",
    "calc_reference_orbit",
    "call",
    "combine_consecutive_elements",
    "debug_marker",
    "end_file",
    "expand_lattice",
    "merge_elements",
    "no_digested",
    "no_superimpose",
    "parameter",
    "parser_debug",
    "particle_start",
    "print",
    "redef",
    "remove_elements",
    "return",
    "root",
    "slice_lattice",
    "start_branch_at",
    "superimpose",
    "title",
    "use",
    "use_local_lat_file",
    "write_digested",
}

BUILTIN_CONSTANTS = {
    "twopi",
    "fourpi",
    "pi",
    "e",
    "e_log",
    "sqrt_2",
    "degrad",
    "degrees",
    "raddeg",
    "m_electron",
    "m_muon",
    "m_pion_0",
    "m_pion_charged",
    "m_proton",
    "m_deuteron",
    "m_neutron",
    "c_light",
    "r_e",
    "r_p",
    "e_charge",
    "h_planck",
    "h_bar_planck",
    "fine_struct_const",
    "anom_moment_electron",
    "anom_moment_proton",
    "anom_moment_neutron",
    "anom_moment_muon",
    "anom_moment_deuteron",
    "anom_moment_he3",
}
