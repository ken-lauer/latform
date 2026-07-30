from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    Does_Not_Exist = "Does_Not_Exist"
    Free = "Free"
    Quasi_Free = "Quasi_Free"
    Dependent = "Dependent"
    Private = "Private"
    Overlay_Slave = "Overlay_Slave"
    Field_Master_Dependent = "Field_Master_Dependent"
    Super_Lord_Align = "Super_Lord_Align"
    Unknown = "Unknown"


class Kind(Enum):
    Real = "Real"
    Integer = "Integer"
    Logical = "Logical"
    Switch = "Switch"
    String = "String"
    Struct = "Struct"
    Unknown = "Unknown"


@dataclass(slots=True, frozen=True)
class Attr:
    name: str
    state: State
    kind: Kind
    units: str
    desc: str = ""


by_element: dict[str, dict[str, Attr]] = {}


by_element["DRIFT"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Private, Kind.Unknown, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Private,
        Kind.Unknown,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Private, Kind.Unknown, "", "If True, apply spin fringe-field kicks."
    ),
    "SPLIT_ID": Attr("SPLIT_ID", State.Private, Kind.Unknown, "", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["SBEND"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", '"Length" of bend.'),
    "ROLL": Attr(
        "ROLL", State.Free, Kind.Real, "rad", "Rotation of the element about the longitudinal axis."
    ),
    "REF_TILT": Attr(
        "REF_TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element and reference orbit about the s axis.",
    ),
    "K1": Attr("K1", State.Quasi_Free, Kind.Real, "1/m^2", "Quadrupole strength."),
    "K2": Attr("K2", State.Quasi_Free, Kind.Real, "1/m^3", "Sextupole strength."),
    "G": Attr("G", State.Quasi_Free, Kind.Real, "1/m", "Design bend strength (= 1/rho)."),
    "DG": Attr(
        "DG", State.Quasi_Free, Kind.Real, "1/m", "Actual - Design bend strength difference."
    ),
    "G_TOT": Attr(
        "G_TOT", State.Dependent, Kind.Real, "1/m", "Net design strength = g + dg Dependent param."
    ),
    "RHO": Attr("RHO", State.Quasi_Free, Kind.Real, "m", "Design bend radius. Dependent param."),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "EXACT_MULTIPOLES": Attr(
        "EXACT_MULTIPOLES",
        State.Free,
        Kind.Switch,
        "",
        "Curved coordinate correction? off is default.",
    ),
    "R0_ELEC": Attr("R0_ELEC", State.Free, Kind.Real, "m", ""),
    "R0_MAG": Attr("R0_MAG", State.Free, Kind.Real, "m", ""),
    "E1": Attr("E1", State.Free, Kind.Real, "rad", "Face angles."),
    "E2": Attr("E2", State.Free, Kind.Real, "rad", "Face angles."),
    "FINT": Attr("FINT", State.Free, Kind.Real, "", "Face field integrals."),
    "FINTX": Attr("FINTX", State.Free, Kind.Real, "", "Face field integrals."),
    "HGAP": Attr("HGAP", State.Free, Kind.Real, "m", "Pole half gap."),
    "HGAPX": Attr("HGAPX", State.Free, Kind.Real, "m", "Pole half gap."),
    "H1": Attr("H1", State.Free, Kind.Real, "1/m", "Face curvature."),
    "H2": Attr("H2", State.Free, Kind.Real, "1/m", "Face curvature."),
    "L_RECTANGLE": Attr("L_RECTANGLE", State.Quasi_Free, Kind.Real, "m", '"Rectangular" length.'),
    "B_FIELD_TOT": Attr(
        "B_FIELD_TOT",
        State.Dependent,
        Kind.Real,
        "T",
        "Net field = b_field + db_field. Dependent param.",
    ),
    "L_SAGITTA": Attr(
        "L_SAGITTA", State.Dependent, Kind.Real, "m", "Sagittal length. Dependent param."
    ),
    "L_CHORD": Attr("L_CHORD", State.Quasi_Free, Kind.Real, "m", "Chord length. See."),
    "FIDUCIAL_PT": Attr("FIDUCIAL_PT", State.Free, Kind.Switch, "", "Default is none."),
    "INIT_NEEDED": Attr("INIT_NEEDED", State.Private, Kind.Unknown, "", ""),
    "ANGLE": Attr("ANGLE", State.Quasi_Free, Kind.Real, "rad", "Design bend angle. Dependent var."),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "B_FIELD": Attr(
        "B_FIELD", State.Quasi_Free, Kind.Real, "T", "Design field strength (= P_0 g / q)."
    ),
    "DB_FIELD": Attr(
        "DB_FIELD", State.Quasi_Free, Kind.Real, "T", "Actual - Design bending field difference."
    ),
    "B1_GRADIENT": Attr(
        "B1_GRADIENT", State.Quasi_Free, Kind.Real, "T/m", "Quadrupole field strength."
    ),
    "B2_GRADIENT": Attr(
        "B2_GRADIENT", State.Quasi_Free, Kind.Real, "T/m^2", "Sextupole field strength."
    ),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "PTC_FRINGE_GEOMETRY": Attr("PTC_FRINGE_GEOMETRY", State.Free, Kind.Switch, "", ""),
    "PTC_FIELD_GEOMETRY": Attr(
        "PTC_FIELD_GEOMETRY", State.Free, Kind.Switch, "", "Default is sector."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "ROLL_TOT": Attr(
        "ROLL_TOT", State.Dependent, Kind.Real, "rad", "Net roll including support misalignments."
    ),
    "REF_TILT_TOT": Attr("REF_TILT_TOT", State.Dependent, Kind.Real, "rad", ""),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["QUADRUPOLE"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "K1": Attr("K1", State.Quasi_Free, Kind.Real, "1/m^2", "Quadrupole strength."),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "R0_ELEC": Attr("R0_ELEC", State.Free, Kind.Real, "m", ""),
    "R0_MAG": Attr("R0_MAG", State.Free, Kind.Real, "m", ""),
    "FQ1": Attr("FQ1", State.Free, Kind.Real, "m", "Soft edge fringe parameter."),
    "FQ2": Attr("FQ2", State.Free, Kind.Real, "m", "Soft edge fringe parameter."),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "B1_GRADIENT": Attr("B1_GRADIENT", State.Quasi_Free, Kind.Real, "T/m", "Field strength."),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["GROUP"] = {
    "INTERPOLATION": Attr("INTERPOLATION", State.Free, Kind.Switch, "", ""),
    "GANG": Attr("GANG", State.Free, Kind.Logical, "", ""),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "X_KNOT": Attr("X_KNOT", State.Free, Kind.Struct, "", ""),
    "Y_KNOT": Attr("Y_KNOT", State.Free, Kind.Real, "", ""),
    "SLAVE": Attr("SLAVE", State.Free, Kind.Real, "", ""),
    "VAR": Attr("VAR", State.Free, Kind.Struct, "", ""),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "ACCORDION_EDGE": Attr(
        "ACCORDION_EDGE", State.Free, Kind.Real, "m", "Element grows or shrinks symmetrically"
    ),
    "START_EDGE": Attr(
        "START_EDGE", State.Free, Kind.String, "", "Varies element's upstream edge s-position"
    ),
    "END_EDGE": Attr(
        "END_EDGE", State.Free, Kind.Real, "m", "Varies element's downstream edge s-position"
    ),
    "S_POSITION": Attr(
        "S_POSITION",
        State.Free,
        Kind.Real,
        "m",
        "Varies element's overall s-position. Constant length.",
    ),
}

by_element["SEXTUPOLE"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "K2": Attr("K2", State.Quasi_Free, Kind.Real, "1/m^3", "Sextupole strength."),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "R0_ELEC": Attr("R0_ELEC", State.Free, Kind.Real, "m", ""),
    "R0_MAG": Attr("R0_MAG", State.Free, Kind.Real, "m", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "B2_GRADIENT": Attr("B2_GRADIENT", State.Quasi_Free, Kind.Real, "T/m^2", "Field strength."),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["OVERLAY"] = {
    "INTERPOLATION": Attr("INTERPOLATION", State.Free, Kind.Switch, "", ""),
    "GANG": Attr("GANG", State.Free, Kind.Logical, "", ""),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "X_KNOT": Attr("X_KNOT", State.Free, Kind.Struct, "", ""),
    "Y_KNOT": Attr("Y_KNOT", State.Free, Kind.Real, "", ""),
    "SLAVE": Attr("SLAVE", State.Free, Kind.Real, "", ""),
    "VAR": Attr("VAR", State.Free, Kind.Struct, "", ""),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
}

by_element["CUSTOM"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "VAL1": Attr("VAL1", State.Free, Kind.Real, "", ""),
    "VAL2": Attr("VAL2", State.Free, Kind.Real, "", ""),
    "VAL3": Attr("VAL3", State.Free, Kind.Real, "", ""),
    "VAL4": Attr("VAL4", State.Free, Kind.Real, "", ""),
    "VAL5": Attr("VAL5", State.Free, Kind.Real, "", ""),
    "VAL6": Attr("VAL6", State.Free, Kind.Real, "", ""),
    "VAL7": Attr("VAL7", State.Free, Kind.Real, "", ""),
    "VAL8": Attr("VAL8", State.Free, Kind.Real, "", ""),
    "VAL9": Attr("VAL9", State.Free, Kind.Real, "", ""),
    "VAL10": Attr("VAL10", State.Free, Kind.Real, "", ""),
    "VAL11": Attr("VAL11", State.Free, Kind.Real, "", ""),
    "VAL12": Attr("VAL12", State.Free, Kind.Real, "", ""),
    "DELTA_E_REF": Attr("DELTA_E_REF", State.Free, Kind.Real, "eV", "Change in energy."),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Quasi_Free,
        Kind.Real,
        "eV",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Quasi_Free,
        Kind.Real,
        "eV",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Dependent, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Dependent, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["TAYLOR"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "DELTA_E_REF": Attr(
        "DELTA_E_REF", State.Free, Kind.Real, "eV", "Change in the reference energy."
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME", State.Free, Kind.Real, "sec", "Change in the reference time."
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "TT<OUT><N1><N2>...": Attr("TT<OUT><N1><N2>...", State.Free, Kind.Unknown, "", ""),
    "X_REF": Attr("X_REF", State.Free, Kind.Real, "m", "$x$ reference orbit component."),
    "PX_REF": Attr("PX_REF", State.Free, Kind.Real, "", "$p_x$ reference orbit component."),
    "Y_REF": Attr("Y_REF", State.Free, Kind.Real, "m", "$y$ reference orbit component."),
    "PY_REF": Attr("PY_REF", State.Free, Kind.Real, "", "$p_y$ reference orbit component."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "Z_REF": Attr("Z_REF", State.Free, Kind.Real, "m", "$z$ reference orbit component."),
    "PZ_REF": Attr("PZ_REF", State.Free, Kind.Real, "", "$p_z$ reference orbit component."),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "REF_ORBIT": Attr("REF_ORBIT", State.Free, Kind.Struct, "", ""),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["RFCAVITY"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "HARMON": Attr("HARMON", State.Quasi_Free, Kind.Real, "", "Harmonic number"),
    "HARMON_MASTER": Attr(
        "HARMON_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "Is harmon or rf_frequency the dependent var with ref energy changes?",
    ),
    "GRADIENT": Attr(
        "GRADIENT",
        State.Dependent,
        Kind.Real,
        "eV/m",
        "Accelerating gradient (V/m). Dependent attribute.",
    ),
    "GRADIENT_ERR": Attr("GRADIENT_ERR", State.Private, Kind.Unknown, "", ""),
    "VOLTAGE": Attr("VOLTAGE", State.Free, Kind.Real, "Volt", "Cavity voltage"),
    "VOLTAGE_ERR": Attr("VOLTAGE_ERR", State.Private, Kind.Unknown, "", ""),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "RF_FREQUENCY": Attr("RF_FREQUENCY", State.Quasi_Free, Kind.Real, "Hz", "Frequency"),
    "RF_WAVELENGTH": Attr("RF_WAVELENGTH", State.Dependent, Kind.Real, "m", ""),
    "KS": Attr("KS", State.Private, Kind.Unknown, "1/m", ""),
    "AUTOSCALE_AMPLITUDE": Attr("AUTOSCALE_AMPLITUDE", State.Free, Kind.Logical, "", ""),
    "AUTOSCALE_PHASE": Attr("AUTOSCALE_PHASE", State.Free, Kind.Logical, "", ""),
    "CAVITY_TYPE": Attr("CAVITY_TYPE", State.Free, Kind.Switch, "", ""),
    "PHI0_MAX": Attr("PHI0_MAX", State.Private, Kind.Unknown, "", ""),
    "PHI0": Attr("PHI0", State.Free, Kind.Real, "rad/2pi", "Cavity phase (rad/2pi)."),
    "PHI0_ERR": Attr("PHI0_ERR", State.Private, Kind.Unknown, "", ""),
    "PHI0_MULTIPASS": Attr(
        "PHI0_MULTIPASS",
        State.Free,
        Kind.Real,
        "rad/2pi",
        "Phase variation with multipass (rad/2pi).",
    ),
    "PHI0_AUTOSCALE": Attr(
        "PHI0_AUTOSCALE",
        State.Quasi_Free,
        Kind.Real,
        "rad/2pi",
        "Set by Bmad if autoscaling is turned on (rad/2pi).",
    ),
    "FIELD_AUTOSCALE": Attr("FIELD_AUTOSCALE", State.Quasi_Free, Kind.Real, "", ""),
    "L_ACTIVE": Attr("L_ACTIVE", State.Dependent, Kind.Real, "m", ""),
    "LONGITUDINAL_MODE": Attr(
        "LONGITUDINAL_MODE",
        State.Free,
        Kind.Integer,
        "",
        "Longitudinal mode. Default is 0. May be 0 or 1.",
    ),
    "N_CELL": Attr("N_CELL", State.Free, Kind.Integer, "", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "N_RF_STEPS": Attr("N_RF_STEPS", State.Free, Kind.Integer, "", ""),
    "COUPLER_PHASE": Attr("COUPLER_PHASE", State.Free, Kind.Real, "rad/2pi", ""),
    "COUPLER_ANGLE": Attr("COUPLER_ANGLE", State.Free, Kind.Real, "rad", ""),
    "COUPLER_STRENGTH": Attr("COUPLER_STRENGTH", State.Free, Kind.Real, "", ""),
    "COUPLER_AT": Attr("COUPLER_AT", State.Free, Kind.Switch, "", ""),
    "BS_FIELD": Attr("BS_FIELD", State.Private, Kind.Unknown, "T", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["ELSEPARATOR"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "VOLTAGE": Attr(
        "VOLTAGE",
        State.Quasi_Free,
        Kind.Real,
        "Volt",
        "Voltage between electrodes. This is a settable dependent variable.",
    ),
    "VOLTAGE_ERR": Attr("VOLTAGE_ERR", State.Private, Kind.Unknown, "", ""),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "R0_ELEC": Attr("R0_ELEC", State.Free, Kind.Real, "m", ""),
    "R0_MAG": Attr("R0_MAG", State.Free, Kind.Real, "m", ""),
    "GAP": Attr("GAP", State.Free, Kind.Real, "", "Distance between electrodes"),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "E_FIELD": Attr(
        "E_FIELD",
        State.Quasi_Free,
        Kind.Real,
        "V/m",
        "Electric field. This is a settable dependent variable.",
    ),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["BEAMBEAM"] = {
    "L": Attr("L", State.Dependent, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "REPETITION_FREQUENCY": Attr(
        "REPETITION_FREQUENCY", State.Free, Kind.Real, "Hz", "Strong beam repetition rate."
    ),
    "S_TWISS_REF": Attr("S_TWISS_REF", State.Free, Kind.Real, "m", ""),
    "BBI_CONSTANT": Attr("BBI_CONSTANT", State.Dependent, Kind.Real, "", "Dependent attribute."),
    "CHARGE": Attr("CHARGE", State.Free, Kind.Real, "", "Strong beam charge. Default = -1"),
    "N_PARTICLE": Attr(
        "N_PARTICLE", State.Free, Kind.Real, "", "Number of particles in strong beam."
    ),
    "SIG_X": Attr(
        "SIG_X", State.Free, Kind.Real, "m", "Horizontal strong beam sigma at the center"
    ),
    "SIG_Y": Attr("SIG_Y", State.Free, Kind.Real, "m", "Vertical strong beam sigma at the center"),
    "SIG_Z": Attr("SIG_Z", State.Free, Kind.Real, "m", "Strong beam length"),
    "KS": Attr("KS", State.Quasi_Free, Kind.Real, "1/m", "Solenoid strength."),
    "N_SLICE": Attr("N_SLICE", State.Free, Kind.Integer, "", "Number of strong beam slices"),
    "BETA_A_STRONG": Attr(
        "BETA_A_STRONG", State.Free, Kind.Real, "m", "Strong beam $a$-mode beta Twiss parameter"
    ),
    "BETA_B_STRONG": Attr(
        "BETA_B_STRONG", State.Free, Kind.Real, "m", "Strong beam $b$-mode beta Twiss parameter"
    ),
    "ALPHA_A_STRONG": Attr(
        "ALPHA_A_STRONG", State.Free, Kind.Real, "", "Strong beam $a$-mode alpha Twiss parameter"
    ),
    "ALPHA_B_STRONG": Attr(
        "ALPHA_B_STRONG", State.Free, Kind.Real, "", "Strong beam $b$-mode alpha Twiss parameter"
    ),
    "SPECIES_STRONG": Attr("SPECIES_STRONG", State.Free, Kind.Unknown, "", "Strong beam species"),
    "E_TOT_STRONG": Attr("E_TOT_STRONG", State.Free, Kind.Real, "eV", ""),
    "PC_STRONG": Attr("PC_STRONG", State.Free, Kind.Real, "eV", "Strong beam momentum."),
    "CMAT_11": Attr("CMAT_11", State.Free, Kind.Real, "", ""),
    "CMAT_12": Attr("CMAT_12", State.Free, Kind.Real, "m", ""),
    "CMAT_21": Attr("CMAT_21", State.Free, Kind.Real, "1/m", ""),
    "CMAT_22": Attr("CMAT_22", State.Free, Kind.Real, "", ""),
    "CROSSING_TIME": Attr(
        "CROSSING_TIME", State.Free, Kind.Real, "sec", "Time when strong beam center reaches IP."
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "CRAB_X1": Attr("CRAB_X1", State.Free, Kind.Real, "", "Crabbing linear coefficient."),
    "CRAB_X2": Attr("CRAB_X2", State.Free, Kind.Real, "1/m", "Crabbing quadratic coefficient."),
    "CRAB_X3": Attr("CRAB_X3", State.Free, Kind.Real, "1/m^2", "Crabbing cubic coefficient."),
    "CRAB_TILT": Attr("CRAB_TILT", State.Free, Kind.Real, "rad", "Crabbing tilt."),
    "CRAB_X4": Attr("CRAB_X4", State.Free, Kind.Real, "1/m^3", "Crabbing 4th order coefficient."),
    "CRAB_X5": Attr("CRAB_X5", State.Free, Kind.Real, "1/m^4", "Crabbing 5th order coefficient."),
    "BS_FIELD": Attr("BS_FIELD", State.Quasi_Free, Kind.Real, "T", "Solenoid field strength."),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["WIGGLER"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "KX": Attr("KX", State.Free, Kind.Real, "1/m", "Planar wiggler horizontal wave number."),
    "B_MAX": Attr(
        "B_MAX",
        State.Free,
        Kind.Real,
        "T",
        "Maximum magnetic field (in T) on the wiggler centerline.",
    ),
    "G_MAX": Attr(
        "G_MAX", State.Dependent, Kind.Real, "1/m", "Maximum bending strength. Dependent attribute."
    ),
    "OSC_AMPLITUDE": Attr(
        "OSC_AMPLITUDE",
        State.Dependent,
        Kind.Real,
        "m",
        "Amplitude of the particle oscillations. Dependent attribute.",
    ),
    "K1X": Attr(
        "K1X",
        State.Dependent,
        Kind.Real,
        "1/m^2",
        "Planar wiggler horizontal defocusing strength. Dep attribute.",
    ),
    "K1Y": Attr(
        "K1Y",
        State.Dependent,
        Kind.Real,
        "1/m^2",
        "Planar wiggler vertical focusing strength. Dep attribute.",
    ),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "R0_ELEC": Attr("R0_ELEC", State.Free, Kind.Real, "m", ""),
    "R0_MAG": Attr("R0_MAG", State.Free, Kind.Real, "m", ""),
    "POLARITY": Attr("POLARITY", State.Free, Kind.Real, "", "For scaling the field."),
    "N_PERIOD": Attr(
        "N_PERIOD", State.Free, Kind.Real, "", "The number of periods. Dependent attribute."
    ),
    "L_PERIOD": Attr(
        "L_PERIOD",
        State.Free,
        Kind.Real,
        "m",
        "Length over which field vector returns to the same orientation.",
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "DELTA_REF_TIME_USER_SET": Attr(
        "DELTA_REF_TIME_USER_SET",
        State.Free,
        Kind.Logical,
        "",
        "Delta_ref_time set in lattice file.",
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME", State.Free, Kind.Real, "sec", "Reference time to traverse the element."
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "TERM": Attr("TERM", State.Free, Kind.Struct, "", ""),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["SOL_QUAD"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "K1": Attr("K1", State.Quasi_Free, Kind.Real, "1/m^2", "Quadrupole strength."),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "R0_ELEC": Attr("R0_ELEC", State.Free, Kind.Real, "m", ""),
    "R0_MAG": Attr("R0_MAG", State.Free, Kind.Real, "m", ""),
    "KS": Attr("KS", State.Quasi_Free, Kind.Real, "1/m", "Solenoid strength."),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "B1_GRADIENT": Attr(
        "B1_GRADIENT", State.Quasi_Free, Kind.Real, "T/m", "Quadrupole Field strength."
    ),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "BS_FIELD": Attr("BS_FIELD", State.Quasi_Free, Kind.Real, "T", "Solenoid Field strength."),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["MARKER"] = {
    "L": Attr("L", State.Dependent, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "X_GAIN_ERR": Attr("X_GAIN_ERR", State.Free, Kind.Real, "m", ""),
    "Y_GAIN_ERR": Attr("Y_GAIN_ERR", State.Free, Kind.Real, "m", ""),
    "CRUNCH": Attr("CRUNCH", State.Free, Kind.Real, "rad", ""),
    "NOISE": Attr("NOISE", State.Free, Kind.Real, "", ""),
    "OSC_AMPLITUDE": Attr("OSC_AMPLITUDE", State.Free, Kind.Real, "m", ""),
    "X_GAIN_CALIB": Attr("X_GAIN_CALIB", State.Free, Kind.Real, "m", ""),
    "Y_GAIN_CALIB": Attr("Y_GAIN_CALIB", State.Free, Kind.Real, "m", ""),
    "CRUNCH_CALIB": Attr("CRUNCH_CALIB", State.Free, Kind.Real, "rad", ""),
    "X_OFFSET_CALIB": Attr("X_OFFSET_CALIB", State.Free, Kind.Real, "m", ""),
    "Y_OFFSET_CALIB": Attr("Y_OFFSET_CALIB", State.Free, Kind.Real, "m", ""),
    "TILT_CALIB": Attr("TILT_CALIB", State.Free, Kind.Real, "rad", ""),
    "DE_ETA_MEAS": Attr("DE_ETA_MEAS", State.Free, Kind.Real, "", ""),
    "N_SAMPLE": Attr("N_SAMPLE", State.Free, Kind.Real, "", ""),
    "X_DISPERSION_ERR": Attr("X_DISPERSION_ERR", State.Free, Kind.Real, "m", ""),
    "Y_DISPERSION_ERR": Attr("Y_DISPERSION_ERR", State.Free, Kind.Real, "m", ""),
    "X_DISPERSION_CALIB": Attr("X_DISPERSION_CALIB", State.Free, Kind.Real, "m", ""),
    "Y_DISPERSION_CALIB": Attr("Y_DISPERSION_CALIB", State.Free, Kind.Real, "m", ""),
    "SPLIT_ID": Attr("SPLIT_ID", State.Private, Kind.Unknown, "", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "P0C_REF_INIT": Attr("P0C_REF_INIT", State.Private, Kind.Unknown, "", ""),
    "E_TOT_REF_INIT": Attr("E_TOT_REF_INIT", State.Private, Kind.Unknown, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "REF_SPECIES": Attr("REF_SPECIES", State.Dependent, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["KICKER"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "H_DISPLACE": Attr("H_DISPLACE", State.Free, Kind.Real, "m", ""),
    "V_DISPLACE": Attr("V_DISPLACE", State.Free, Kind.Real, "m^3", ""),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "R0_ELEC": Attr("R0_ELEC", State.Free, Kind.Real, "m", ""),
    "R0_MAG": Attr("R0_MAG", State.Free, Kind.Real, "m", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["HYBRID"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "DELTA_E_REF": Attr("DELTA_E_REF", State.Free, Kind.Real, "eV", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME", State.Free, Kind.Real, "sec", "Reference time to traverse the element."
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["OCTUPOLE"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "K3": Attr("K3", State.Quasi_Free, Kind.Real, "1/m^4", "Octupole strength."),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "R0_ELEC": Attr("R0_ELEC", State.Free, Kind.Real, "m", ""),
    "R0_MAG": Attr("R0_MAG", State.Free, Kind.Real, "m", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "B3_GRADIENT": Attr("B3_GRADIENT", State.Quasi_Free, Kind.Real, "T/m^3", "Field strength."),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["RBEND"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", '"Length" of bend.'),
    "ROLL": Attr(
        "ROLL", State.Free, Kind.Real, "rad", "Rotation of the element about the longitudinal axis."
    ),
    "REF_TILT": Attr(
        "REF_TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element and reference orbit about the s axis.",
    ),
    "K1": Attr("K1", State.Quasi_Free, Kind.Real, "1/m^2", "Quadrupole strength."),
    "K2": Attr("K2", State.Quasi_Free, Kind.Real, "1/m^3", "Sextupole strength."),
    "G": Attr("G", State.Quasi_Free, Kind.Real, "1/m", "Design bend strength (= 1/rho)."),
    "DG": Attr(
        "DG", State.Quasi_Free, Kind.Real, "1/m", "Actual - Design bend strength difference."
    ),
    "G_TOT": Attr(
        "G_TOT", State.Dependent, Kind.Real, "1/m", "Net design strength = g + dg Dependent param."
    ),
    "RHO": Attr("RHO", State.Quasi_Free, Kind.Real, "m", "Design bend radius. Dependent param."),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "EXACT_MULTIPOLES": Attr(
        "EXACT_MULTIPOLES",
        State.Free,
        Kind.Switch,
        "",
        "Curved coordinate correction? off is default.",
    ),
    "R0_ELEC": Attr("R0_ELEC", State.Free, Kind.Real, "m", ""),
    "R0_MAG": Attr("R0_MAG", State.Free, Kind.Real, "m", ""),
    "E1": Attr("E1", State.Free, Kind.Real, "rad", "Face angles."),
    "E2": Attr("E2", State.Free, Kind.Real, "rad", "Face angles."),
    "FINT": Attr("FINT", State.Free, Kind.Real, "", "Face field integrals."),
    "FINTX": Attr("FINTX", State.Free, Kind.Real, "", "Face field integrals."),
    "HGAP": Attr("HGAP", State.Free, Kind.Real, "m", "Pole half gap."),
    "HGAPX": Attr("HGAPX", State.Free, Kind.Real, "m", "Pole half gap."),
    "H1": Attr("H1", State.Free, Kind.Real, "1/m", "Face curvature."),
    "H2": Attr("H2", State.Free, Kind.Real, "1/m", "Face curvature."),
    "L_RECTANGLE": Attr("L_RECTANGLE", State.Quasi_Free, Kind.Real, "m", '"Rectangular" length.'),
    "B_FIELD_TOT": Attr(
        "B_FIELD_TOT",
        State.Dependent,
        Kind.Real,
        "T",
        "Net field = b_field + db_field. Dependent param.",
    ),
    "L_SAGITTA": Attr(
        "L_SAGITTA", State.Dependent, Kind.Real, "m", "Sagittal length. Dependent param."
    ),
    "L_CHORD": Attr("L_CHORD", State.Quasi_Free, Kind.Real, "m", "Chord length. See."),
    "FIDUCIAL_PT": Attr("FIDUCIAL_PT", State.Free, Kind.Switch, "", "Default is none."),
    "INIT_NEEDED": Attr("INIT_NEEDED", State.Private, Kind.Unknown, "", ""),
    "ANGLE": Attr("ANGLE", State.Quasi_Free, Kind.Real, "rad", "Design bend angle. Dependent var."),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "B_FIELD": Attr(
        "B_FIELD", State.Quasi_Free, Kind.Real, "T", "Design field strength (= P_0 g / q)."
    ),
    "DB_FIELD": Attr(
        "DB_FIELD", State.Quasi_Free, Kind.Real, "T", "Actual - Design bending field difference."
    ),
    "B1_GRADIENT": Attr(
        "B1_GRADIENT", State.Quasi_Free, Kind.Real, "T/m", "Quadrupole field strength."
    ),
    "B2_GRADIENT": Attr(
        "B2_GRADIENT", State.Quasi_Free, Kind.Real, "T/m^2", "Sextupole field strength."
    ),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "PTC_FRINGE_GEOMETRY": Attr("PTC_FRINGE_GEOMETRY", State.Free, Kind.Switch, "", ""),
    "PTC_FIELD_GEOMETRY": Attr(
        "PTC_FIELD_GEOMETRY", State.Free, Kind.Switch, "", "Default is sector."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "ROLL_TOT": Attr(
        "ROLL_TOT", State.Dependent, Kind.Real, "rad", "Net roll including support misalignments."
    ),
    "REF_TILT_TOT": Attr("REF_TILT_TOT", State.Dependent, Kind.Real, "rad", ""),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["MULTIPOLE"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "K0L_STATUS": Attr("K0L_STATUS", State.Free, Kind.Switch, "", ""),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "K0L": Attr(
        "K0L", State.Free, Kind.Real, "", "Integrated normal multipole strength of order 0."
    ),
    "K1L": Attr(
        "K1L", State.Free, Kind.Real, "1/m", "Integrated normal multipole strength of order 1."
    ),
    "K2L": Attr(
        "K2L", State.Free, Kind.Real, "1/m^2", "Integrated normal multipole strength of order 2."
    ),
    "K3L": Attr(
        "K3L", State.Free, Kind.Real, "1/m^3", "Integrated normal multipole strength of order 3."
    ),
    "K4L": Attr(
        "K4L", State.Free, Kind.Real, "1/m^4", "Integrated normal multipole strength of order 4."
    ),
    "K5L": Attr(
        "K5L", State.Free, Kind.Real, "1/m^5", "Integrated normal multipole strength of order 5."
    ),
    "K6L": Attr(
        "K6L", State.Free, Kind.Real, "1/m^6", "Integrated normal multipole strength of order 6."
    ),
    "K7L": Attr(
        "K7L", State.Free, Kind.Real, "1/m^7", "Integrated normal multipole strength of order 7."
    ),
    "K8L": Attr(
        "K8L", State.Free, Kind.Real, "1/m^8", "Integrated normal multipole strength of order 8."
    ),
    "K9L": Attr(
        "K9L", State.Free, Kind.Real, "1/m^9", "Integrated normal multipole strength of order 9."
    ),
    "K10L": Attr(
        "K10L", State.Free, Kind.Real, "1/m^10", "Integrated normal multipole strength of order 10."
    ),
    "K11L": Attr(
        "K11L", State.Free, Kind.Real, "1/m^11", "Integrated normal multipole strength of order 11."
    ),
    "K12L": Attr(
        "K12L", State.Free, Kind.Real, "1/m^12", "Integrated normal multipole strength of order 12."
    ),
    "K13L": Attr(
        "K13L", State.Free, Kind.Real, "1/m^13", "Integrated normal multipole strength of order 13."
    ),
    "K14L": Attr(
        "K14L", State.Free, Kind.Real, "1/m^14", "Integrated normal multipole strength of order 14."
    ),
    "K15L": Attr(
        "K15L", State.Free, Kind.Real, "1/m^15", "Integrated normal multipole strength of order 15."
    ),
    "K16L": Attr(
        "K16L", State.Free, Kind.Real, "1/m^16", "Integrated normal multipole strength of order 16."
    ),
    "K17L": Attr(
        "K17L", State.Free, Kind.Real, "1/m^17", "Integrated normal multipole strength of order 17."
    ),
    "K18L": Attr(
        "K18L", State.Free, Kind.Real, "1/m^18", "Integrated normal multipole strength of order 18."
    ),
    "K19L": Attr(
        "K19L", State.Free, Kind.Real, "1/m^19", "Integrated normal multipole strength of order 19."
    ),
    "K20L": Attr(
        "K20L", State.Free, Kind.Real, "1/m^20", "Integrated normal multipole strength of order 20."
    ),
    "K21L": Attr(
        "K21L", State.Free, Kind.Real, "1/m^21", "Integrated normal multipole strength of order 21."
    ),
    "T0": Attr("T0", State.Free, Kind.Real, "rad", "Tilt of the order-0 multipole."),
    "T1": Attr("T1", State.Free, Kind.Real, "rad", "Tilt of the order-1 multipole."),
    "T2": Attr("T2", State.Free, Kind.Real, "rad", "Tilt of the order-2 multipole."),
    "T3": Attr("T3", State.Free, Kind.Real, "rad", "Tilt of the order-3 multipole."),
    "T4": Attr("T4", State.Free, Kind.Real, "rad", "Tilt of the order-4 multipole."),
    "T5": Attr("T5", State.Free, Kind.Real, "rad", "Tilt of the order-5 multipole."),
    "T6": Attr("T6", State.Free, Kind.Real, "rad", "Tilt of the order-6 multipole."),
    "T7": Attr("T7", State.Free, Kind.Real, "rad", "Tilt of the order-7 multipole."),
    "T8": Attr("T8", State.Free, Kind.Real, "rad", "Tilt of the order-8 multipole."),
    "T9": Attr("T9", State.Free, Kind.Real, "rad", "Tilt of the order-9 multipole."),
    "T10": Attr("T10", State.Free, Kind.Real, "rad", "Tilt of the order-10 multipole."),
    "T11": Attr("T11", State.Free, Kind.Real, "rad", "Tilt of the order-11 multipole."),
    "T12": Attr("T12", State.Free, Kind.Real, "rad", "Tilt of the order-12 multipole."),
    "T13": Attr("T13", State.Free, Kind.Real, "rad", "Tilt of the order-13 multipole."),
    "T14": Attr("T14", State.Free, Kind.Real, "rad", "Tilt of the order-14 multipole."),
    "T15": Attr("T15", State.Free, Kind.Real, "rad", "Tilt of the order-15 multipole."),
    "T16": Attr("T16", State.Free, Kind.Real, "rad", "Tilt of the order-16 multipole."),
    "T17": Attr("T17", State.Free, Kind.Real, "rad", "Tilt of the order-17 multipole."),
    "T18": Attr("T18", State.Free, Kind.Real, "rad", "Tilt of the order-18 multipole."),
    "T19": Attr("T19", State.Free, Kind.Real, "rad", "Tilt of the order-19 multipole."),
    "T20": Attr("T20", State.Free, Kind.Real, "rad", "Tilt of the order-20 multipole."),
    "T21": Attr("T21", State.Free, Kind.Real, "rad", "Tilt of the order-21 multipole."),
}

by_element["!MAD_BEAM"] = {
    "N_PART": Attr("N_PART", State.Free, Kind.Real, "", ""),
    "PC": Attr("PC", State.Free, Kind.Real, "eV", ""),
    "ENERGY": Attr("ENERGY", State.Free, Kind.Real, "eV", ""),
    "PARTICLE": Attr("PARTICLE", State.Free, Kind.Unknown, "", ""),
}

by_element["AB_MULTIPOLE"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["SOLENOID"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "R_SOLENOID": Attr("R_SOLENOID", State.Free, Kind.Real, "m", "Solenoid radius."),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "R0_ELEC": Attr("R0_ELEC", State.Free, Kind.Real, "m", ""),
    "R0_MAG": Attr("R0_MAG", State.Free, Kind.Real, "m", ""),
    "KS": Attr("KS", State.Quasi_Free, Kind.Real, "1/m", "Normalized solenoid strength."),
    "L_SOFT_EDGE": Attr("L_SOFT_EDGE", State.Free, Kind.Real, "m", 'For modeling a "soft" fringe.'),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "BS_FIELD": Attr("BS_FIELD", State.Quasi_Free, Kind.Real, "T", "Solenoid field strength."),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["PATCH"] = {
    "L": Attr("L", State.Quasi_Free, Kind.Real, "m", "Reference length."),
    "TILT": Attr("TILT", State.Free, Kind.Real, "rad", "Exit face orientation from Entrance."),
    "REF_COORDS": Attr(
        "REF_COORDS", State.Free, Kind.Switch, "", "Coordinate system defining the length."
    ),
    "FLEXIBLE": Attr("FLEXIBLE", State.Free, Kind.Logical, "", "Default: False."),
    "USER_SETS_LENGTH": Attr(
        "USER_SETS_LENGTH", State.Free, Kind.Logical, "", "User sets element length? Default is F."
    ),
    "UPSTREAM_ELE_DIR": Attr("UPSTREAM_ELE_DIR", State.Dependent, Kind.Integer, "", ""),
    "DOWNSTREAM_ELE_DIR": Attr("DOWNSTREAM_ELE_DIR", State.Dependent, Kind.Integer, "", ""),
    "T_OFFSET": Attr("T_OFFSET", State.Free, Kind.Real, "sec", "Reference time offset."),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Exit face orientation from Entrance."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Exit face orientation from Entrance."
    ),
    "X_OFFSET": Attr("X_OFFSET", State.Free, Kind.Real, "m", "Exit face offset from Entrance."),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Exit face offset from Entrance."),
    "Z_OFFSET": Attr("Z_OFFSET", State.Free, Kind.Real, "m", "Exit face offset from Entrance."),
    "E_TOT_OFFSET": Attr("E_TOT_OFFSET", State.Free, Kind.Real, "eV", ""),
    "E_TOT_SET": Attr("E_TOT_SET", State.Free, Kind.Real, "eV", ""),
    "P0C_SET": Attr("P0C_SET", State.Free, Kind.Real, "eV", "Reference momentum at exit end (eV)."),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Dependent,
        Kind.Real,
        "eV",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Dependent,
        Kind.Real,
        "eV",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["LCAVITY"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "WARN_COUNT": Attr("WARN_COUNT", State.Private, Kind.Unknown, "", ""),
    "GRADIENT_TOT": Attr(
        "GRADIENT_TOT",
        State.Dependent,
        Kind.Real,
        "eV/m",
        "Net gradient = gradient + gradient_err. Dependent param.",
    ),
    "GRADIENT": Attr("GRADIENT", State.Free, Kind.Real, "eV/m", "Accelerating gradient (V/m)."),
    "GRADIENT_ERR": Attr(
        "GRADIENT_ERR", State.Free, Kind.Real, "eV/m", "Accelerating gradient error (V/m)."
    ),
    "VOLTAGE": Attr(
        "VOLTAGE", State.Quasi_Free, Kind.Real, "Volt", "Cavity voltage. Dependent attribute."
    ),
    "VOLTAGE_ERR": Attr("VOLTAGE_ERR", State.Quasi_Free, Kind.Real, "Volt", "Error voltage"),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "RF_FREQUENCY": Attr("RF_FREQUENCY", State.Free, Kind.Real, "Hz", "RF frequency (Hz)."),
    "RF_WAVELENGTH": Attr("RF_WAVELENGTH", State.Dependent, Kind.Real, "m", ""),
    "KS": Attr("KS", State.Quasi_Free, Kind.Real, "1/m", "Normalized solenoid strength."),
    "AUTOSCALE_AMPLITUDE": Attr("AUTOSCALE_AMPLITUDE", State.Free, Kind.Logical, "", ""),
    "AUTOSCALE_PHASE": Attr("AUTOSCALE_PHASE", State.Free, Kind.Logical, "", ""),
    "E_LOSS": Attr(
        "E_LOSS", State.Free, Kind.Real, "eV", "Loss parameter for short range wakefields (V/Coul)."
    ),
    "CAVITY_TYPE": Attr("CAVITY_TYPE", State.Free, Kind.Switch, "", "Type of cavity."),
    "PHI0": Attr(
        "PHI0",
        State.Free,
        Kind.Real,
        "rad/2pi",
        "Phase (rad/2\\(\\pi\\)) of the reference particle with",
    ),
    "PHI0_ERR": Attr("PHI0_ERR", State.Free, Kind.Real, "rad/2pi", "Phase error (rad/2\\(\\pi\\))"),
    "PHI0_MULTIPASS": Attr(
        "PHI0_MULTIPASS",
        State.Free,
        Kind.Real,
        "rad/2pi",
        "Phase (rad/2\\(\\pi\\)) with respect to a multipass lord.",
    ),
    "PHI0_AUTOSCALE": Attr(
        "PHI0_AUTOSCALE",
        State.Quasi_Free,
        Kind.Real,
        "rad/2pi",
        "Set by Bmad when autoscaling is turned on.",
    ),
    "FIELD_AUTOSCALE": Attr(
        "FIELD_AUTOSCALE",
        State.Quasi_Free,
        Kind.Real,
        "",
        "Set by Bmad when autoscaling is turned on.",
    ),
    "VOLTAGE_TOT": Attr(
        "VOLTAGE_TOT",
        State.Dependent,
        Kind.Real,
        "Volt",
        "Net voltage = voltage + voltage_err. Dependent param.",
    ),
    "L_ACTIVE": Attr(
        "L_ACTIVE", State.Dependent, Kind.Real, "m", "Active region length. Dependent attribute."
    ),
    "LONGITUDINAL_MODE": Attr(
        "LONGITUDINAL_MODE",
        State.Free,
        Kind.Integer,
        "",
        "Longitudinal mode. Default is 0. May be 0 or 1.",
    ),
    "N_CELL": Attr(
        "N_CELL", State.Free, Kind.Integer, "", "Number of cavity cells. Default is -1."
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "N_RF_STEPS": Attr(
        "N_RF_STEPS", State.Free, Kind.Integer, "", "Number of steps in kick-drift-kick model."
    ),
    "COUPLER_PHASE": Attr("COUPLER_PHASE", State.Free, Kind.Real, "rad/2pi", ""),
    "COUPLER_ANGLE": Attr("COUPLER_ANGLE", State.Free, Kind.Real, "rad", ""),
    "COUPLER_STRENGTH": Attr("COUPLER_STRENGTH", State.Free, Kind.Real, "", ""),
    "COUPLER_AT": Attr("COUPLER_AT", State.Free, Kind.Switch, "", ""),
    "BS_FIELD": Attr("BS_FIELD", State.Quasi_Free, Kind.Real, "T", "Solenoid field strength."),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Quasi_Free,
        Kind.Real,
        "eV",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Quasi_Free,
        Kind.Real,
        "eV",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Dependent, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Dependent, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["!PARAMETER"] = {
    "N_PART": Attr("N_PART", State.Free, Kind.Real, "", ""),
    "TAYLOR_ORDER": Attr("TAYLOR_ORDER", State.Free, Kind.Integer, "", ""),
    "IX_BRANCH": Attr("IX_BRANCH", State.Private, Kind.Unknown, "", ""),
    "DEFAULT_TRACKING_SPECIES": Attr("DEFAULT_TRACKING_SPECIES", State.Free, Kind.Unknown, "", ""),
    "HIGH_ENERGY_SPACE_CHARGE_ON": Attr(
        "HIGH_ENERGY_SPACE_CHARGE_ON", State.Free, Kind.Logical, "", ""
    ),
    "PHOTON_TYPE": Attr("PHOTON_TYPE", State.Free, Kind.Switch, "", ""),
    "LATTICE_TYPE": Attr("LATTICE_TYPE", State.Free, Kind.Switch, "", ""),
    "LIVE_BRANCH": Attr("LIVE_BRANCH", State.Free, Kind.Logical, "", ""),
    "GEOMETRY": Attr("GEOMETRY", State.Free, Kind.Switch, "", ""),
    "P0C": Attr(
        "P0C", State.Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "NO_END_MARKER": Attr("NO_END_MARKER", State.Free, Kind.Logical, "", ""),
    "ELECTRIC_DIPOLE_MOMENT": Attr("ELECTRIC_DIPOLE_MOMENT", State.Free, Kind.Real, "", ""),
    "PARSER_MAKE_XFER_MATS": Attr("PARSER_MAKE_XFER_MATS", State.Free, Kind.Logical, "", ""),
    "LATTICE": Attr("LATTICE", State.Free, Kind.String, "", ""),
    "MACHINE": Attr("MACHINE", State.Free, Kind.String, "", ""),
    "ABSOLUTE_TIME_TRACKING": Attr("ABSOLUTE_TIME_TRACKING", State.Free, Kind.Logical, "", ""),
    "PTC_EXACT_MISALIGN": Attr("PTC_EXACT_MISALIGN", State.Free, Kind.Logical, "", ""),
    "PTC_EXACT_MODEL": Attr("PTC_EXACT_MODEL", State.Free, Kind.Logical, "", ""),
    "RAN_SEED": Attr("RAN_SEED", State.Free, Kind.Real, "", ""),
    "PARTICLE": Attr("PARTICLE", State.Free, Kind.Unknown, "", ""),
}

by_element["NULL_ELE"] = {
    "IX_BRANCH": Attr("IX_BRANCH", State.Private, Kind.Unknown, "", ""),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["BEGINNING_ELE"] = {
    "L": Attr("L", State.Private, Kind.Unknown, "", "Length of the element."),
    "INHERIT_FROM_FORK": Attr("INHERIT_FROM_FORK", State.Free, Kind.Logical, "", ""),
    "DETA_DS_MASTER": Attr("DETA_DS_MASTER", State.Private, Kind.Unknown, "", ""),
    "IX_FIXER": Attr("IX_FIXER", State.Private, Kind.Unknown, "", ""),
    "SPIN_DN_DPZ_X": Attr("SPIN_DN_DPZ_X", State.Quasi_Free, Kind.Real, "", ""),
    "SPIN_DN_DPZ_Y": Attr("SPIN_DN_DPZ_Y", State.Quasi_Free, Kind.Real, "", ""),
    "SPIN_DN_DPZ_Z": Attr("SPIN_DN_DPZ_Z", State.Quasi_Free, Kind.Real, "", ""),
    "X_STORED": Attr("X_STORED", State.Free, Kind.Real, "m", ""),
    "PX_STORED": Attr("PX_STORED", State.Free, Kind.Real, "", ""),
    "Y_STORED": Attr("Y_STORED", State.Free, Kind.Real, "m", ""),
    "PY_STORED": Attr("PY_STORED", State.Free, Kind.Real, "", ""),
    "Z_STORED": Attr("Z_STORED", State.Free, Kind.Real, "m", ""),
    "PZ_STORED": Attr("PZ_STORED", State.Free, Kind.Real, "", ""),
    "BETA_A_STORED": Attr("BETA_A_STORED", State.Free, Kind.Real, "m", ""),
    "ALPHA_A_STORED": Attr("ALPHA_A_STORED", State.Free, Kind.Real, "", ""),
    "BETA_B_STORED": Attr("BETA_B_STORED", State.Free, Kind.Real, "m", ""),
    "ALPHA_B_STORED": Attr("ALPHA_B_STORED", State.Free, Kind.Real, "", ""),
    "PHI_A_STORED": Attr("PHI_A_STORED", State.Free, Kind.Real, "", ""),
    "PHI_B_STORED": Attr("PHI_B_STORED", State.Free, Kind.Real, "", ""),
    "MODE_FLIP_STORED": Attr("MODE_FLIP_STORED", State.Free, Kind.Logical, "", ""),
    "CMAT_11": Attr("CMAT_11", State.Quasi_Free, Kind.Real, "", ""),
    "CMAT_12": Attr("CMAT_12", State.Quasi_Free, Kind.Real, "m", ""),
    "CMAT_21": Attr("CMAT_21", State.Quasi_Free, Kind.Real, "1/m", ""),
    "CMAT_22": Attr("CMAT_22", State.Quasi_Free, Kind.Real, "", ""),
    "MODE_FLIP": Attr("MODE_FLIP", State.Quasi_Free, Kind.Logical, "", ""),
    "ETA_X_STORED": Attr("ETA_X_STORED", State.Free, Kind.Real, "m", ""),
    "ETAP_X_STORED": Attr("ETAP_X_STORED", State.Free, Kind.Real, "", ""),
    "ETA_Y_STORED": Attr("ETA_Y_STORED", State.Free, Kind.Real, "m", ""),
    "ETAP_Y_STORED": Attr("ETAP_Y_STORED", State.Free, Kind.Real, "", ""),
    "CMAT_11_STORED": Attr("CMAT_11_STORED", State.Free, Kind.Real, "", ""),
    "CMAT_12_STORED": Attr("CMAT_12_STORED", State.Free, Kind.Real, "m", ""),
    "CMAT_21_STORED": Attr("CMAT_21_STORED", State.Free, Kind.Real, "1/m", ""),
    "CMAT_22_STORED": Attr("CMAT_22_STORED", State.Free, Kind.Real, "", ""),
    "DBETA_DPZ_A_STORED": Attr("DBETA_DPZ_A_STORED", State.Free, Kind.Real, "m", ""),
    "DBETA_DPZ_B_STORED": Attr("DBETA_DPZ_B_STORED", State.Free, Kind.Real, "m", ""),
    "DALPHA_DPZ_A_STORED": Attr("DALPHA_DPZ_A_STORED", State.Free, Kind.Real, "", ""),
    "DALPHA_DPZ_B_STORED": Attr("DALPHA_DPZ_B_STORED", State.Free, Kind.Real, "", ""),
    "DETA_DPZ_X_STORED": Attr("DETA_DPZ_X_STORED", State.Free, Kind.Real, "m", ""),
    "DETA_DPZ_Y_STORED": Attr("DETA_DPZ_Y_STORED", State.Free, Kind.Real, "m", ""),
    "DETAP_DPZ_X_STORED": Attr("DETAP_DPZ_X_STORED", State.Free, Kind.Real, "", ""),
    "DETAP_DPZ_Y_STORED": Attr("DETAP_DPZ_Y_STORED", State.Free, Kind.Real, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME", State.Private, Kind.Unknown, "", "Reference time to traverse the element."
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Dependent,
        Kind.Real,
        "eV",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Dependent,
        Kind.Real,
        "eV",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "SPIN_X_STORED": Attr("SPIN_X_STORED", State.Free, Kind.Real, "", ""),
    "SPIN_Y_STORED": Attr("SPIN_Y_STORED", State.Free, Kind.Real, "", ""),
    "SPIN_Z_STORED": Attr("SPIN_Z_STORED", State.Free, Kind.Real, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Private, Kind.Unknown, "", "Reference time at the entrance end."
    ),
    "DCMAT_DPZ_11_STORED": Attr("DCMAT_DPZ_11_STORED", State.Free, Kind.Real, "", ""),
    "DCMAT_DPZ_12_STORED": Attr("DCMAT_DPZ_12_STORED", State.Free, Kind.Real, "", ""),
    "DCMAT_DPZ_21_STORED": Attr("DCMAT_DPZ_21_STORED", State.Free, Kind.Real, "", ""),
    "DCMAT_DPZ_22_STORED": Attr("DCMAT_DPZ_22_STORED", State.Free, Kind.Real, "", ""),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ETA_X": Attr("ETA_X", State.Quasi_Free, Kind.Real, "m", ""),
    "ETA_Y": Attr("ETA_Y", State.Quasi_Free, Kind.Real, "m", ""),
    "ETAP_X": Attr("ETAP_X", State.Quasi_Free, Kind.Real, "", ""),
    "ETAP_Y": Attr("ETAP_Y", State.Quasi_Free, Kind.Real, "", ""),
    "PHI_A": Attr("PHI_A", State.Quasi_Free, Kind.Real, "rad", ""),
    "ETA_Z": Attr("ETA_Z", State.Quasi_Free, Kind.Real, "m", ""),
    "DETA_DPZ_X": Attr("DETA_DPZ_X", State.Quasi_Free, Kind.Real, "m", ""),
    "DETAP_DPZ_X": Attr("DETAP_DPZ_X", State.Quasi_Free, Kind.Real, "", ""),
    "REF_TIME": Attr("REF_TIME", State.Quasi_Free, Kind.Real, "sec", ""),
    "DETA_DPZ_Y": Attr("DETA_DPZ_Y", State.Quasi_Free, Kind.Real, "m", ""),
    "DETAP_DPZ_Y": Attr("DETAP_DPZ_Y", State.Quasi_Free, Kind.Real, "", ""),
    "ALPHA_A": Attr("ALPHA_A", State.Quasi_Free, Kind.Real, "", ""),
    "ALPHA_B": Attr("ALPHA_B", State.Quasi_Free, Kind.Real, "", ""),
    "S": Attr("S", State.Quasi_Free, Kind.Real, "m", ""),
    "X_POSITION": Attr("X_POSITION", State.Quasi_Free, Kind.Real, "m", ""),
    "Y_POSITION": Attr("Y_POSITION", State.Quasi_Free, Kind.Real, "m", ""),
    "Z_POSITION": Attr("Z_POSITION", State.Quasi_Free, Kind.Real, "m", ""),
    "THETA_POSITION": Attr("THETA_POSITION", State.Quasi_Free, Kind.Real, "rad", ""),
    "PHI_POSITION": Attr("PHI_POSITION", State.Quasi_Free, Kind.Real, "rad", ""),
    "PSI_POSITION": Attr("PSI_POSITION", State.Quasi_Free, Kind.Real, "rad", ""),
    "BETA_A": Attr("BETA_A", State.Quasi_Free, Kind.Real, "m", ""),
    "BETA_B": Attr("BETA_B", State.Quasi_Free, Kind.Real, "m", ""),
    "DBETA_DPZ_A": Attr("DBETA_DPZ_A", State.Quasi_Free, Kind.Real, "m", ""),
    "DBETA_DPZ_B": Attr("DBETA_DPZ_B", State.Quasi_Free, Kind.Real, "m", ""),
    "DALPHA_DPZ_A": Attr("DALPHA_DPZ_A", State.Quasi_Free, Kind.Real, "", ""),
    "DALPHA_DPZ_B": Attr("DALPHA_DPZ_B", State.Quasi_Free, Kind.Real, "", ""),
    "PHI_B": Attr("PHI_B", State.Quasi_Free, Kind.Real, "rad", ""),
    "REF_SPECIES": Attr("REF_SPECIES", State.Free, Kind.Unknown, "", ""),
}

by_element["!LINE"] = {
    "L": Attr("L", State.Private, Kind.Unknown, "", "Length of the element."),
    "INHERIT_FROM_FORK": Attr("INHERIT_FROM_FORK", State.Free, Kind.Logical, "", ""),
    "DETA_DS_MASTER": Attr("DETA_DS_MASTER", State.Private, Kind.Unknown, "", ""),
    "IX_FIXER": Attr("IX_FIXER", State.Private, Kind.Unknown, "", ""),
    "IX_BRANCH": Attr("IX_BRANCH", State.Private, Kind.Unknown, "", ""),
    "SPIN_DN_DPZ_X": Attr("SPIN_DN_DPZ_X", State.Quasi_Free, Kind.Real, "", ""),
    "SPIN_DN_DPZ_Y": Attr("SPIN_DN_DPZ_Y", State.Quasi_Free, Kind.Real, "", ""),
    "SPIN_DN_DPZ_Z": Attr("SPIN_DN_DPZ_Z", State.Quasi_Free, Kind.Real, "", ""),
    "DEFAULT_TRACKING_SPECIES": Attr("DEFAULT_TRACKING_SPECIES", State.Free, Kind.Unknown, "", ""),
    "CMAT_11": Attr("CMAT_11", State.Quasi_Free, Kind.Real, "", ""),
    "CMAT_12": Attr("CMAT_12", State.Quasi_Free, Kind.Real, "m", ""),
    "CMAT_21": Attr("CMAT_21", State.Quasi_Free, Kind.Real, "1/m", ""),
    "CMAT_22": Attr("CMAT_22", State.Quasi_Free, Kind.Real, "", ""),
    "MODE_FLIP": Attr("MODE_FLIP", State.Quasi_Free, Kind.Logical, "", ""),
    "HIGH_ENERGY_SPACE_CHARGE_ON": Attr(
        "HIGH_ENERGY_SPACE_CHARGE_ON", State.Free, Kind.Logical, "", ""
    ),
    "LIVE_BRANCH": Attr("LIVE_BRANCH", State.Free, Kind.Logical, "", ""),
    "GEOMETRY": Attr("GEOMETRY", State.Free, Kind.Switch, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME", State.Private, Kind.Unknown, "", "Reference time to traverse the element."
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Dependent,
        Kind.Real,
        "eV",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Dependent,
        Kind.Real,
        "eV",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Private, Kind.Unknown, "", "Reference time at the entrance end."
    ),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ETA_X": Attr("ETA_X", State.Quasi_Free, Kind.Real, "m", ""),
    "ETA_Y": Attr("ETA_Y", State.Quasi_Free, Kind.Real, "m", ""),
    "ETAP_X": Attr("ETAP_X", State.Quasi_Free, Kind.Real, "", ""),
    "ETAP_Y": Attr("ETAP_Y", State.Quasi_Free, Kind.Real, "", ""),
    "PHI_A": Attr("PHI_A", State.Quasi_Free, Kind.Real, "rad", ""),
    "ETA_Z": Attr("ETA_Z", State.Quasi_Free, Kind.Real, "m", ""),
    "DETA_DPZ_X": Attr("DETA_DPZ_X", State.Quasi_Free, Kind.Real, "m", ""),
    "DETAP_DPZ_X": Attr("DETAP_DPZ_X", State.Quasi_Free, Kind.Real, "", ""),
    "REF_TIME": Attr("REF_TIME", State.Quasi_Free, Kind.Real, "sec", ""),
    "DETA_DPZ_Y": Attr("DETA_DPZ_Y", State.Quasi_Free, Kind.Real, "m", ""),
    "DETAP_DPZ_Y": Attr("DETAP_DPZ_Y", State.Quasi_Free, Kind.Real, "", ""),
    "ALPHA_A": Attr("ALPHA_A", State.Quasi_Free, Kind.Real, "", ""),
    "ALPHA_B": Attr("ALPHA_B", State.Quasi_Free, Kind.Real, "", ""),
    "S": Attr("S", State.Quasi_Free, Kind.Real, "m", ""),
    "X_POSITION": Attr("X_POSITION", State.Quasi_Free, Kind.Real, "m", ""),
    "Y_POSITION": Attr("Y_POSITION", State.Quasi_Free, Kind.Real, "m", ""),
    "Z_POSITION": Attr("Z_POSITION", State.Quasi_Free, Kind.Real, "m", ""),
    "THETA_POSITION": Attr("THETA_POSITION", State.Quasi_Free, Kind.Real, "rad", ""),
    "PHI_POSITION": Attr("PHI_POSITION", State.Quasi_Free, Kind.Real, "rad", ""),
    "PSI_POSITION": Attr("PSI_POSITION", State.Quasi_Free, Kind.Real, "rad", ""),
    "BETA_A": Attr("BETA_A", State.Quasi_Free, Kind.Real, "m", ""),
    "BETA_B": Attr("BETA_B", State.Quasi_Free, Kind.Real, "m", ""),
    "DBETA_DPZ_A": Attr("DBETA_DPZ_A", State.Quasi_Free, Kind.Real, "m", ""),
    "DBETA_DPZ_B": Attr("DBETA_DPZ_B", State.Quasi_Free, Kind.Real, "m", ""),
    "DALPHA_DPZ_A": Attr("DALPHA_DPZ_A", State.Quasi_Free, Kind.Real, "", ""),
    "DALPHA_DPZ_B": Attr("DALPHA_DPZ_B", State.Quasi_Free, Kind.Real, "", ""),
    "PHI_B": Attr("PHI_B", State.Quasi_Free, Kind.Real, "rad", ""),
    "PARTICLE": Attr("PARTICLE", State.Free, Kind.Unknown, "", ""),
}

by_element["MATCH"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "BETA_A0": Attr("BETA_A0", State.Free, Kind.Real, "m", "Entrance betas"),
    "ALPHA_A0": Attr("ALPHA_A0", State.Free, Kind.Real, "", "Entrance alphas"),
    "BETA_B0": Attr("BETA_B0", State.Free, Kind.Real, "m", "Entrance betas"),
    "ALPHA_B0": Attr("ALPHA_B0", State.Free, Kind.Real, "", "Entrance alphas"),
    "BETA_A1": Attr("BETA_A1", State.Free, Kind.Real, "m", "Exit betas"),
    "ALPHA_A1": Attr("ALPHA_A1", State.Free, Kind.Real, "", "Exit alphas"),
    "BETA_B1": Attr("BETA_B1", State.Free, Kind.Real, "m", "Exit betas"),
    "ALPHA_B1": Attr("ALPHA_B1", State.Free, Kind.Real, "", "Exit alphas"),
    "DPHI_A": Attr("DPHI_A", State.Free, Kind.Real, "rad", "Phase advances (radians)."),
    "DPHI_B": Attr("DPHI_B", State.Free, Kind.Real, "rad", "Phase advances (radians)."),
    "ETA_X0": Attr("ETA_X0", State.Free, Kind.Real, "m", "Entrance etas"),
    "ETAP_X0": Attr("ETAP_X0", State.Free, Kind.Real, "", "Entrance momentum dispersion"),
    "ETA_Y0": Attr("ETA_Y0", State.Free, Kind.Real, "m", "Entrance etas"),
    "ETAP_Y0": Attr("ETAP_Y0", State.Free, Kind.Real, "", "Entrance momentum dispersion"),
    "ETA_X1": Attr("ETA_X1", State.Free, Kind.Real, "m", "Exit etas"),
    "ETAP_X1": Attr("ETAP_X1", State.Free, Kind.Real, "", "Exit eta'"),
    "ETA_Y1": Attr("ETA_Y1", State.Free, Kind.Real, "m", "Exit etas"),
    "ETAP_Y1": Attr("ETAP_Y1", State.Free, Kind.Real, "", "Exit eta'"),
    "C11_MAT0": Attr("C11_MAT0", State.Free, Kind.Real, "", "Entrance coupling."),
    "C12_MAT0": Attr("C12_MAT0", State.Free, Kind.Real, "m", "Entrance coupling."),
    "C21_MAT0": Attr("C21_MAT0", State.Free, Kind.Real, "1/m", "Entrance coupling."),
    "C22_MAT0": Attr("C22_MAT0", State.Free, Kind.Real, "", "Entrance coupling."),
    "MODE_FLIP0": Attr(
        "MODE_FLIP0", State.Free, Kind.Logical, "", "Mode flip status. Default: False."
    ),
    "C11_MAT1": Attr("C11_MAT1", State.Free, Kind.Real, "", "Exit coupling."),
    "C12_MAT1": Attr("C12_MAT1", State.Free, Kind.Real, "m", "Exit coupling."),
    "C21_MAT1": Attr("C21_MAT1", State.Free, Kind.Real, "1/m", "Exit coupling."),
    "C22_MAT1": Attr("C22_MAT1", State.Free, Kind.Real, "", "Exit coupling."),
    "MODE_FLIP1": Attr(
        "MODE_FLIP1", State.Free, Kind.Logical, "", "Mode flip status. Default: False."
    ),
    "X0": Attr("X0", State.Free, Kind.Real, "m", "Entrance coordinates"),
    "PX0": Attr("PX0", State.Free, Kind.Real, "", "Entrance coordinates"),
    "Y0": Attr("Y0", State.Free, Kind.Real, "m", "Entrance coordinates"),
    "PY0": Attr("PY0", State.Free, Kind.Real, "", "Entrance coordinates"),
    "Z0": Attr("Z0", State.Free, Kind.Real, "m", "Entrance coordinates"),
    "PZ0": Attr("PZ0", State.Free, Kind.Real, "", "Entrance coordinates"),
    "X1": Attr("X1", State.Free, Kind.Real, "m", "Exit coordinates"),
    "PX1": Attr("PX1", State.Free, Kind.Real, "", "Exit coordinates"),
    "Y1": Attr("Y1", State.Free, Kind.Real, "m", "Exit coordinates"),
    "PY1": Attr("PY1", State.Free, Kind.Real, "", "Exit coordinates"),
    "Z1": Attr("Z1", State.Free, Kind.Real, "m", "Exit coordinates"),
    "PZ1": Attr("PZ1", State.Free, Kind.Real, "", "Exit coordinates"),
    "MATRIX": Attr("MATRIX", State.Free, Kind.Switch, "", "Matrix calculation. Default: standard."),
    "KICK0": Attr("KICK0", State.Free, Kind.Switch, "", "Zeroth order calc. Default: standard."),
    "RECALC": Attr(
        "RECALC", State.Free, Kind.Logical, "", "Calculate transfer map? Default is True."
    ),
    "DELTA_TIME": Attr("DELTA_TIME", State.Free, Kind.Real, "sec", "Change in time."),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["MONITOR"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "X_GAIN_ERR": Attr("X_GAIN_ERR", State.Free, Kind.Real, "m", ""),
    "Y_GAIN_ERR": Attr("Y_GAIN_ERR", State.Free, Kind.Real, "m", ""),
    "CRUNCH": Attr("CRUNCH", State.Free, Kind.Real, "rad", ""),
    "NOISE": Attr("NOISE", State.Free, Kind.Real, "", ""),
    "OSC_AMPLITUDE": Attr("OSC_AMPLITUDE", State.Free, Kind.Real, "m", ""),
    "X_GAIN_CALIB": Attr("X_GAIN_CALIB", State.Free, Kind.Real, "m", ""),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "Y_GAIN_CALIB": Attr("Y_GAIN_CALIB", State.Free, Kind.Real, "m", ""),
    "CRUNCH_CALIB": Attr("CRUNCH_CALIB", State.Free, Kind.Real, "rad", ""),
    "X_OFFSET_CALIB": Attr("X_OFFSET_CALIB", State.Free, Kind.Real, "m", ""),
    "Y_OFFSET_CALIB": Attr("Y_OFFSET_CALIB", State.Free, Kind.Real, "m", ""),
    "TILT_CALIB": Attr("TILT_CALIB", State.Free, Kind.Real, "rad", ""),
    "DE_ETA_MEAS": Attr("DE_ETA_MEAS", State.Free, Kind.Real, "", ""),
    "N_SAMPLE": Attr("N_SAMPLE", State.Free, Kind.Real, "", ""),
    "X_DISPERSION_ERR": Attr("X_DISPERSION_ERR", State.Free, Kind.Real, "m", ""),
    "Y_DISPERSION_ERR": Attr("Y_DISPERSION_ERR", State.Free, Kind.Real, "m", ""),
    "X_DISPERSION_CALIB": Attr("X_DISPERSION_CALIB", State.Free, Kind.Real, "m", ""),
    "Y_DISPERSION_CALIB": Attr("Y_DISPERSION_CALIB", State.Free, Kind.Real, "m", ""),
    "SPLIT_ID": Attr("SPLIT_ID", State.Private, Kind.Unknown, "", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["INSTRUMENT"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "X_GAIN_ERR": Attr("X_GAIN_ERR", State.Free, Kind.Real, "m", ""),
    "Y_GAIN_ERR": Attr("Y_GAIN_ERR", State.Free, Kind.Real, "m", ""),
    "CRUNCH": Attr("CRUNCH", State.Free, Kind.Real, "rad", ""),
    "NOISE": Attr("NOISE", State.Free, Kind.Real, "", ""),
    "OSC_AMPLITUDE": Attr("OSC_AMPLITUDE", State.Free, Kind.Real, "m", ""),
    "X_GAIN_CALIB": Attr("X_GAIN_CALIB", State.Free, Kind.Real, "m", ""),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "Y_GAIN_CALIB": Attr("Y_GAIN_CALIB", State.Free, Kind.Real, "m", ""),
    "CRUNCH_CALIB": Attr("CRUNCH_CALIB", State.Free, Kind.Real, "rad", ""),
    "X_OFFSET_CALIB": Attr("X_OFFSET_CALIB", State.Free, Kind.Real, "m", ""),
    "Y_OFFSET_CALIB": Attr("Y_OFFSET_CALIB", State.Free, Kind.Real, "m", ""),
    "TILT_CALIB": Attr("TILT_CALIB", State.Free, Kind.Real, "rad", ""),
    "DE_ETA_MEAS": Attr("DE_ETA_MEAS", State.Free, Kind.Real, "", ""),
    "N_SAMPLE": Attr("N_SAMPLE", State.Free, Kind.Real, "", ""),
    "X_DISPERSION_ERR": Attr("X_DISPERSION_ERR", State.Free, Kind.Real, "m", ""),
    "Y_DISPERSION_ERR": Attr("Y_DISPERSION_ERR", State.Free, Kind.Real, "m", ""),
    "X_DISPERSION_CALIB": Attr("X_DISPERSION_CALIB", State.Free, Kind.Real, "m", ""),
    "Y_DISPERSION_CALIB": Attr("Y_DISPERSION_CALIB", State.Free, Kind.Real, "m", ""),
    "SPLIT_ID": Attr("SPLIT_ID", State.Private, Kind.Unknown, "", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["HKICKER"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "KICK": Attr("KICK", State.Quasi_Free, Kind.Real, "", "Kick strength."),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "BL_KICK": Attr("BL_KICK", State.Quasi_Free, Kind.Real, "T*m", ""),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["VKICKER"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "KICK": Attr("KICK", State.Quasi_Free, Kind.Real, "", "Kick strength."),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "BL_KICK": Attr("BL_KICK", State.Quasi_Free, Kind.Real, "T*m", ""),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["RCOLLIMATOR"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "PX_APERTURE_WIDTH2": Attr(
        "PX_APERTURE_WIDTH2", State.Free, Kind.Real, "", "Px aperture half width"
    ),
    "PX_APERTURE_CENTER": Attr(
        "PX_APERTURE_CENTER", State.Free, Kind.Real, "", "Px aperture center"
    ),
    "PY_APERTURE_WIDTH2": Attr(
        "PY_APERTURE_WIDTH2", State.Free, Kind.Real, "", "Py aperture half width"
    ),
    "PY_APERTURE_CENTER": Attr(
        "PY_APERTURE_CENTER", State.Free, Kind.Real, "", "Py aperture center"
    ),
    "Z_APERTURE_WIDTH2": Attr(
        "Z_APERTURE_WIDTH2", State.Free, Kind.Real, "m", "Z aperture half width"
    ),
    "Z_APERTURE_CENTER": Attr("Z_APERTURE_CENTER", State.Free, Kind.Real, "m", "Z aperture center"),
    "PZ_APERTURE_WIDTH2": Attr(
        "PZ_APERTURE_WIDTH2", State.Free, Kind.Real, "", "Pz aperture half width"
    ),
    "PZ_APERTURE_CENTER": Attr(
        "PZ_APERTURE_CENTER", State.Free, Kind.Real, "", "Pz aperture center"
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["ECOLLIMATOR"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "PX_APERTURE_WIDTH2": Attr(
        "PX_APERTURE_WIDTH2", State.Free, Kind.Real, "", "Px aperture half width"
    ),
    "PX_APERTURE_CENTER": Attr(
        "PX_APERTURE_CENTER", State.Free, Kind.Real, "", "Px aperture center"
    ),
    "PY_APERTURE_WIDTH2": Attr(
        "PY_APERTURE_WIDTH2", State.Free, Kind.Real, "", "Py aperture half width"
    ),
    "PY_APERTURE_CENTER": Attr(
        "PY_APERTURE_CENTER", State.Free, Kind.Real, "", "Py aperture center"
    ),
    "Z_APERTURE_WIDTH2": Attr(
        "Z_APERTURE_WIDTH2", State.Free, Kind.Real, "m", "Z aperture half width"
    ),
    "Z_APERTURE_CENTER": Attr("Z_APERTURE_CENTER", State.Free, Kind.Real, "m", "Z aperture center"),
    "PZ_APERTURE_WIDTH2": Attr(
        "PZ_APERTURE_WIDTH2", State.Free, Kind.Real, "", "Pz aperture half width"
    ),
    "PZ_APERTURE_CENTER": Attr(
        "PZ_APERTURE_CENTER", State.Free, Kind.Real, "", "Pz aperture center"
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["GIRDER"] = {
    "L": Attr("L", State.Dependent, Kind.Real, "m", 'Girder "Length". Dependent attribute.'),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "REF_TILT": Attr(
        "REF_TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element and reference orbit about the s axis.",
    ),
    "ORIGIN_ELE_REF_PT": Attr(
        "ORIGIN_ELE_REF_PT", State.Free, Kind.Switch, "", "Reference pt on reference ele."
    ),
    "DX_ORIGIN": Attr("DX_ORIGIN", State.Free, Kind.Real, "m", "X-position offset"),
    "DY_ORIGIN": Attr("DY_ORIGIN", State.Free, Kind.Real, "m", "Y-position offset"),
    "DZ_ORIGIN": Attr("DZ_ORIGIN", State.Free, Kind.Real, "m", "Z-position offset"),
    "DTHETA_ORIGIN": Attr(
        "DTHETA_ORIGIN", State.Free, Kind.Real, "rad", "Orientation angle offset."
    ),
    "DPHI_ORIGIN": Attr("DPHI_ORIGIN", State.Free, Kind.Real, "rad", "Orientation angle offset."),
    "DPSI_ORIGIN": Attr("DPSI_ORIGIN", State.Free, Kind.Real, "rad", "Orientation angle offset."),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT", State.Free, Kind.Real, "rad", "Net x pitch including support misalignments."
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT", State.Free, Kind.Real, "rad", "Net y pitch including support misalignments."
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Free,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Free,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Free,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Free, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "REF_TILT_TOT": Attr("REF_TILT_TOT", State.Free, Kind.Real, "rad", ""),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "ORIGIN_ELE": Attr("ORIGIN_ELE", State.Free, Kind.String, "", "Reference element."),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
}

by_element["CONVERTER"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "PC_OUT_MIN": Attr(
        "PC_OUT_MIN", State.Free, Kind.Real, "eV", "Minimum outgoing particle momentum (eV)."
    ),
    "PC_OUT_MAX": Attr(
        "PC_OUT_MAX", State.Free, Kind.Real, "eV", "Maximum outgoing particle momentum (eV)."
    ),
    "ANGLE_OUT_MAX": Attr("ANGLE_OUT_MAX", State.Free, Kind.Real, "rad", "Maximum outgoing angle."),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Dependent,
        Kind.Real,
        "eV",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Dependent,
        Kind.Real,
        "eV",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr("P0C", State.Free, Kind.Real, "eV", "Output ref momentum."),
    "E_TOT": Attr(
        "E_TOT", State.Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "DISTRIBUTION": Attr(
        "DISTRIBUTION", State.Free, Kind.Switch, "", "Outgoing particle distribution."
    ),
    "SPECIES_OUT": Attr("SPECIES_OUT", State.Free, Kind.Unknown, "", "Output species."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["!PARTICLE_START"] = {
    "X": Attr("X", State.Free, Kind.Real, "m", ""),
    "PX": Attr("PX", State.Free, Kind.Real, "", ""),
    "Y": Attr("Y", State.Free, Kind.Real, "m", ""),
    "PY": Attr("PY", State.Free, Kind.Real, "", ""),
    "Z": Attr("Z", State.Free, Kind.Real, "m", ""),
    "PZ": Attr("PZ", State.Free, Kind.Real, "", ""),
    "T": Attr("T", State.Free, Kind.Real, "sec", ""),
    "E_PHOTON": Attr("E_PHOTON", State.Free, Kind.Real, "eV", ""),
    "FIELD_X": Attr("FIELD_X", State.Free, Kind.Real, "", ""),
    "FIELD_Y": Attr("FIELD_Y", State.Free, Kind.Real, "", ""),
    "PHASE_X": Attr("PHASE_X", State.Free, Kind.Real, "rad", ""),
    "PHASE_Y": Attr("PHASE_Y", State.Free, Kind.Real, "rad", ""),
    "SIG_Z": Attr("SIG_Z", State.Free, Kind.Real, "m", ""),
    "SIG_PZ": Attr("SIG_PZ", State.Free, Kind.Real, "", ""),
    "SPIN_X": Attr("SPIN_X", State.Free, Kind.Real, "", ""),
    "SPIN_Y": Attr("SPIN_Y", State.Free, Kind.Real, "", ""),
    "SPIN_Z": Attr("SPIN_Z", State.Free, Kind.Real, "", ""),
    "EMITTANCE_A": Attr("EMITTANCE_A", State.Free, Kind.Real, "m*rad", ""),
    "EMITTANCE_B": Attr("EMITTANCE_B", State.Free, Kind.Real, "m*rad", ""),
    "EMITTANCE_Z": Attr("EMITTANCE_Z", State.Free, Kind.Real, "m*rad", ""),
}

by_element["PHOTON_FORK"] = {
    "L": Attr("L", State.Dependent, Kind.Real, "m", "Length of the element."),
    "DIRECTION": Attr(
        "DIRECTION", State.Free, Kind.Integer, "", "Particles are entering or leaving?"
    ),
    "NEW_BRANCH": Attr(
        "NEW_BRANCH",
        State.Free,
        Kind.Logical,
        "",
        "Make a new branch from the to_line? Default = True.",
    ),
    "IX_TO_BRANCH": Attr("IX_TO_BRANCH", State.Dependent, Kind.Integer, "", ""),
    "IX_TO_ELEMENT": Attr("IX_TO_ELEMENT", State.Dependent, Kind.Integer, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "TO_LINE": Attr("TO_LINE", State.Free, Kind.String, "", "What line to fork to."),
    "TO_ELEMENT": Attr(
        "TO_ELEMENT",
        State.Free,
        Kind.String,
        "",
        "What element to attach to in the line being forked to.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "REF_SPECIES": Attr("REF_SPECIES", State.Dependent, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["FORK"] = {
    "L": Attr("L", State.Dependent, Kind.Unknown, "m", "Length of the element."),
    "DIRECTION": Attr(
        "DIRECTION", State.Free, Kind.Integer, "", "Particles are entering or leaving?"
    ),
    "NEW_BRANCH": Attr(
        "NEW_BRANCH",
        State.Free,
        Kind.Logical,
        "",
        "Make a new branch from the to_line? Default = True.",
    ),
    "IX_TO_BRANCH": Attr("IX_TO_BRANCH", State.Dependent, Kind.Integer, "", ""),
    "IX_TO_ELEMENT": Attr("IX_TO_ELEMENT", State.Dependent, Kind.Integer, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "TO_LINE": Attr("TO_LINE", State.Free, Kind.String, "", "What line to fork to."),
    "TO_ELEMENT": Attr(
        "TO_ELEMENT",
        State.Free,
        Kind.String,
        "",
        "What element to attach to in the line being forked to.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "REF_SPECIES": Attr("REF_SPECIES", State.Dependent, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["MIRROR"] = {
    "L": Attr("L", State.Dependent, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "REF_TILT": Attr(
        "REF_TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element and reference orbit about the s axis.",
    ),
    "GRAZE_ANGLE": Attr(
        "GRAZE_ANGLE",
        State.Free,
        Kind.Real,
        "rad",
        "Angle between incoming beam and mirror surface.",
    ),
    "CRITICAL_ANGLE": Attr("CRITICAL_ANGLE", State.Free, Kind.Real, "rad", "Critical angle."),
    "USE_REFLECTIVITY_TABLE": Attr("USE_REFLECTIVITY_TABLE", State.Free, Kind.Logical, "", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "REF_TILT_TOT": Attr("REF_TILT_TOT", State.Dependent, Kind.Real, "rad", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "REF_WAVELENGTH": Attr(
        "REF_WAVELENGTH",
        State.Dependent,
        Kind.Real,
        "m",
        "Reference wavelength. Dependent attribute.",
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "DISPLACEMENT": Attr("DISPLACEMENT", State.Free, Kind.Struct, "", ""),
    "SEGMENTED": Attr("SEGMENTED", State.Free, Kind.Struct, "", ""),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "REFLECTIVITY_TABLE": Attr("REFLECTIVITY_TABLE", State.Free, Kind.Struct, "", ""),
    "CURVATURE": Attr("CURVATURE", State.Free, Kind.Struct, "", ""),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "CURVATURE_X0_Y2": Attr("CURVATURE_X0_Y2", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X0_Y3": Attr("CURVATURE_X0_Y3", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X0_Y4": Attr("CURVATURE_X0_Y4", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X0_Y5": Attr("CURVATURE_X0_Y5", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X0_Y6": Attr("CURVATURE_X0_Y6", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X1_Y1": Attr("CURVATURE_X1_Y1", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X1_Y2": Attr("CURVATURE_X1_Y2", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X1_Y3": Attr("CURVATURE_X1_Y3", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X1_Y4": Attr("CURVATURE_X1_Y4", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X1_Y5": Attr("CURVATURE_X1_Y5", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X2_Y0": Attr("CURVATURE_X2_Y0", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X2_Y1": Attr("CURVATURE_X2_Y1", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X2_Y2": Attr("CURVATURE_X2_Y2", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X2_Y3": Attr("CURVATURE_X2_Y3", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X2_Y4": Attr("CURVATURE_X2_Y4", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X3_Y0": Attr("CURVATURE_X3_Y0", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X3_Y1": Attr("CURVATURE_X3_Y1", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X3_Y2": Attr("CURVATURE_X3_Y2", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X3_Y3": Attr("CURVATURE_X3_Y3", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X4_Y0": Attr("CURVATURE_X4_Y0", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X4_Y1": Attr("CURVATURE_X4_Y1", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X4_Y2": Attr("CURVATURE_X4_Y2", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X5_Y0": Attr("CURVATURE_X5_Y0", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X5_Y1": Attr("CURVATURE_X5_Y1", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X6_Y0": Attr("CURVATURE_X6_Y0", State.Free, Kind.Real, "1/m^5", ""),
}

by_element["CRYSTAL"] = {
    "L": Attr("L", State.Dependent, Kind.Real, "m", "Length of reference orbit."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "REF_TILT": Attr(
        "REF_TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element and reference orbit about the s axis.",
    ),
    "TILT_CORR": Attr(
        "TILT_CORR", State.Dependent, Kind.Real, "rad", "Tilt correction due to a finite psi_angle."
    ),
    "REF_ORBIT_FOLLOWS": Attr(
        "REF_ORBIT_FOLLOWS",
        State.Free,
        Kind.Switch,
        "",
        "Reference orbit aligned with what outgoing beam?",
    ),
    "BRAGG_ANGLE_IN": Attr(
        "BRAGG_ANGLE_IN",
        State.Dependent,
        Kind.Real,
        "rad",
        "Incoming grazing angle for Bragg diffraction.",
    ),
    "BRAGG_ANGLE_OUT": Attr(
        "BRAGG_ANGLE_OUT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Outgoing grazing angle for Bragg diffraction.",
    ),
    "BRAGG_ANGLE": Attr(
        "BRAGG_ANGLE",
        State.Dependent,
        Kind.Real,
        "rad",
        "Nominal Bragg angle at the reference wave length.",
    ),
    "DBRAGG_ANGLE_DE": Attr(
        "DBRAGG_ANGLE_DE",
        State.Dependent,
        Kind.Real,
        "rad/eV",
        "Variation of the Bragg angle with energy (radians/eV).",
    ),
    "DARWIN_WIDTH_SIGMA": Attr(
        "DARWIN_WIDTH_SIGMA",
        State.Dependent,
        Kind.Real,
        "rad",
        "Darwin width for sigma polarized light (radians).",
    ),
    "DARWIN_WIDTH_PI": Attr(
        "DARWIN_WIDTH_PI",
        State.Dependent,
        Kind.Real,
        "rad",
        "Darwin width for pi polarized light (radians).",
    ),
    "PENDELLOSUNG_PERIOD_SIGMA": Attr(
        "PENDELLOSUNG_PERIOD_SIGMA",
        State.Dependent,
        Kind.Real,
        "m",
        "Pendellosung period for sigma polarized light.",
    ),
    "PENDELLOSUNG_PERIOD_PI": Attr(
        "PENDELLOSUNG_PERIOD_PI",
        State.Dependent,
        Kind.Real,
        "m",
        "Pendellosung period for pi polarized light.",
    ),
    "GRAZE_ANGLE_IN": Attr(
        "GRAZE_ANGLE_IN",
        State.Free,
        Kind.Real,
        "rad",
        "Angle between incoming ref orbit and surface.",
    ),
    "GRAZE_ANGLE_OUT": Attr(
        "GRAZE_ANGLE_OUT",
        State.Free,
        Kind.Real,
        "rad",
        "Angle between outgoing ref orbit and surface.",
    ),
    "ALPHA_ANGLE": Attr(
        "ALPHA_ANGLE",
        State.Dependent,
        Kind.Real,
        "",
        "Angle of H-vector with respect to the surface normal.",
    ),
    "PSI_ANGLE": Attr(
        "PSI_ANGLE", State.Free, Kind.Real, "rad", "Rotation of H-vector about the surface normal."
    ),
    "V_UNITCELL": Attr("V_UNITCELL", State.Dependent, Kind.Real, "m^3", "Unit cell volume."),
    "IS_MOSAIC": Attr("IS_MOSAIC", State.Free, Kind.Logical, "", ""),
    "MOSAIC_THICKNESS": Attr("MOSAIC_THICKNESS", State.Free, Kind.Real, "m", ""),
    "MOSAIC_ANGLE_RMS_IN_PLANE": Attr(
        "MOSAIC_ANGLE_RMS_IN_PLANE", State.Free, Kind.Real, "rad", ""
    ),
    "MOSAIC_ANGLE_RMS_OUT_PLANE": Attr(
        "MOSAIC_ANGLE_RMS_OUT_PLANE", State.Free, Kind.Real, "rad", ""
    ),
    "MOSAIC_DIFFRACTION_NUM": Attr("MOSAIC_DIFFRACTION_NUM", State.Free, Kind.Integer, "", ""),
    "B_PARAM": Attr(
        "B_PARAM", State.Free, Kind.Real, "", "B parameter for photons with the reference energy."
    ),
    "REF_CAP_GAMMA": Attr(
        "REF_CAP_GAMMA",
        State.Dependent,
        Kind.Real,
        "",
        "\\(\\Gamma\\) at the reference wavelength.",
    ),
    "USE_REFLECTIVITY_TABLE": Attr("USE_REFLECTIVITY_TABLE", State.Free, Kind.Logical, "", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "D_SPACING": Attr("D_SPACING", State.Dependent, Kind.Real, "m", "Lattice plane spacing."),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "REF_TILT_TOT": Attr("REF_TILT_TOT", State.Dependent, Kind.Real, "rad", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "THICKNESS": Attr(
        "THICKNESS", State.Free, Kind.Real, "m", "Thickness of crystal for Laue diffraction."
    ),
    "REF_WAVELENGTH": Attr(
        "REF_WAVELENGTH",
        State.Dependent,
        Kind.Real,
        "m",
        "Reference wavelength. Dependent attribute.",
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "H_MISALIGN": Attr("H_MISALIGN", State.Free, Kind.Struct, "", ""),
    "DISPLACEMENT": Attr("DISPLACEMENT", State.Free, Kind.Struct, "", ""),
    "SEGMENTED": Attr("SEGMENTED", State.Free, Kind.Struct, "", ""),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "REFLECTIVITY_TABLE": Attr("REFLECTIVITY_TABLE", State.Free, Kind.Struct, "", ""),
    "CURVATURE": Attr("CURVATURE", State.Free, Kind.Struct, "", ""),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "CRYSTAL_TYPE": Attr(
        "CRYSTAL_TYPE", State.Free, Kind.String, "", "Crystal material and reflection plane."
    ),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "CURVATURE_X0_Y2": Attr("CURVATURE_X0_Y2", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X0_Y3": Attr("CURVATURE_X0_Y3", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X0_Y4": Attr("CURVATURE_X0_Y4", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X0_Y5": Attr("CURVATURE_X0_Y5", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X0_Y6": Attr("CURVATURE_X0_Y6", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X1_Y1": Attr("CURVATURE_X1_Y1", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X1_Y2": Attr("CURVATURE_X1_Y2", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X1_Y3": Attr("CURVATURE_X1_Y3", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X1_Y4": Attr("CURVATURE_X1_Y4", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X1_Y5": Attr("CURVATURE_X1_Y5", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X2_Y0": Attr("CURVATURE_X2_Y0", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X2_Y1": Attr("CURVATURE_X2_Y1", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X2_Y2": Attr("CURVATURE_X2_Y2", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X2_Y3": Attr("CURVATURE_X2_Y3", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X2_Y4": Attr("CURVATURE_X2_Y4", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X3_Y0": Attr("CURVATURE_X3_Y0", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X3_Y1": Attr("CURVATURE_X3_Y1", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X3_Y2": Attr("CURVATURE_X3_Y2", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X3_Y3": Attr("CURVATURE_X3_Y3", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X4_Y0": Attr("CURVATURE_X4_Y0", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X4_Y1": Attr("CURVATURE_X4_Y1", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X4_Y2": Attr("CURVATURE_X4_Y2", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X5_Y0": Attr("CURVATURE_X5_Y0", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X5_Y1": Attr("CURVATURE_X5_Y1", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X6_Y0": Attr("CURVATURE_X6_Y0", State.Free, Kind.Real, "1/m^5", ""),
}

by_element["PIPE"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "X_GAIN_ERR": Attr("X_GAIN_ERR", State.Free, Kind.Real, "m", ""),
    "Y_GAIN_ERR": Attr("Y_GAIN_ERR", State.Free, Kind.Real, "m", ""),
    "CRUNCH": Attr("CRUNCH", State.Free, Kind.Real, "rad", ""),
    "NOISE": Attr("NOISE", State.Free, Kind.Real, "", ""),
    "OSC_AMPLITUDE": Attr("OSC_AMPLITUDE", State.Free, Kind.Real, "m", ""),
    "X_GAIN_CALIB": Attr("X_GAIN_CALIB", State.Free, Kind.Real, "m", ""),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "Y_GAIN_CALIB": Attr("Y_GAIN_CALIB", State.Free, Kind.Real, "m", ""),
    "CRUNCH_CALIB": Attr("CRUNCH_CALIB", State.Free, Kind.Real, "rad", ""),
    "X_OFFSET_CALIB": Attr("X_OFFSET_CALIB", State.Free, Kind.Real, "m", ""),
    "Y_OFFSET_CALIB": Attr("Y_OFFSET_CALIB", State.Free, Kind.Real, "m", ""),
    "TILT_CALIB": Attr("TILT_CALIB", State.Free, Kind.Real, "rad", ""),
    "DE_ETA_MEAS": Attr("DE_ETA_MEAS", State.Free, Kind.Real, "", ""),
    "N_SAMPLE": Attr("N_SAMPLE", State.Free, Kind.Real, "", ""),
    "X_DISPERSION_ERR": Attr("X_DISPERSION_ERR", State.Free, Kind.Real, "m", ""),
    "Y_DISPERSION_ERR": Attr("Y_DISPERSION_ERR", State.Free, Kind.Real, "m", ""),
    "X_DISPERSION_CALIB": Attr("X_DISPERSION_CALIB", State.Free, Kind.Real, "m", ""),
    "Y_DISPERSION_CALIB": Attr("Y_DISPERSION_CALIB", State.Free, Kind.Real, "m", ""),
    "SPLIT_ID": Attr("SPLIT_ID", State.Private, Kind.Unknown, "", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["CAPILLARY"] = {
    "L": Attr("L", State.Dependent, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "CRITICAL_ANGLE_FACTOR": Attr(
        "CRITICAL_ANGLE_FACTOR",
        State.Free,
        Kind.Real,
        "rad*eV",
        "Critical angle * Energy (rad * eV)",
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "N_SLICE_SPLINE": Attr("N_SLICE_SPLINE", State.Free, Kind.Real, "", ""),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["MULTILAYER_MIRROR"] = {
    "L": Attr("L", State.Dependent, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "REF_TILT": Attr(
        "REF_TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element and reference orbit about the s axis.",
    ),
    "GRAZE_ANGLE": Attr(
        "GRAZE_ANGLE",
        State.Free,
        Kind.Real,
        "rad",
        "Angle between incoming beam and mirror surface.",
    ),
    "D1_THICKNESS": Attr("D1_THICKNESS", State.Free, Kind.Real, "m", "Thickness of layer 1"),
    "D2_THICKNESS": Attr("D2_THICKNESS", State.Free, Kind.Real, "m", "Thickness of layer 2"),
    "V1_UNITCELL": Attr(
        "V1_UNITCELL", State.Free, Kind.Real, "m^3", "Unit cell volume for layer 1"
    ),
    "V2_UNITCELL": Attr(
        "V2_UNITCELL", State.Free, Kind.Real, "m^3", "Unit cell volume for layer 2"
    ),
    "N_CELL": Attr(
        "N_CELL", State.Free, Kind.Integer, "", "Number of cells (= Number of layers / 2)"
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "REF_TILT_TOT": Attr("REF_TILT_TOT", State.Dependent, Kind.Real, "rad", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "REF_WAVELENGTH": Attr(
        "REF_WAVELENGTH",
        State.Dependent,
        Kind.Real,
        "m",
        "Reference wavelength. Dependent attribute.",
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "DISPLACEMENT": Attr("DISPLACEMENT", State.Free, Kind.Struct, "", ""),
    "SEGMENTED": Attr("SEGMENTED", State.Free, Kind.Struct, "", ""),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "CURVATURE": Attr("CURVATURE", State.Free, Kind.Struct, "", ""),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "MATERIAL_TYPE": Attr("MATERIAL_TYPE", State.Free, Kind.String, "", "Materials in each layer."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "CURVATURE_X0_Y2": Attr("CURVATURE_X0_Y2", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X0_Y3": Attr("CURVATURE_X0_Y3", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X0_Y4": Attr("CURVATURE_X0_Y4", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X0_Y5": Attr("CURVATURE_X0_Y5", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X0_Y6": Attr("CURVATURE_X0_Y6", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X1_Y1": Attr("CURVATURE_X1_Y1", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X1_Y2": Attr("CURVATURE_X1_Y2", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X1_Y3": Attr("CURVATURE_X1_Y3", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X1_Y4": Attr("CURVATURE_X1_Y4", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X1_Y5": Attr("CURVATURE_X1_Y5", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X2_Y0": Attr("CURVATURE_X2_Y0", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X2_Y1": Attr("CURVATURE_X2_Y1", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X2_Y2": Attr("CURVATURE_X2_Y2", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X2_Y3": Attr("CURVATURE_X2_Y3", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X2_Y4": Attr("CURVATURE_X2_Y4", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X3_Y0": Attr("CURVATURE_X3_Y0", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X3_Y1": Attr("CURVATURE_X3_Y1", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X3_Y2": Attr("CURVATURE_X3_Y2", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X3_Y3": Attr("CURVATURE_X3_Y3", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X4_Y0": Attr("CURVATURE_X4_Y0", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X4_Y1": Attr("CURVATURE_X4_Y1", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X4_Y2": Attr("CURVATURE_X4_Y2", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X5_Y0": Attr("CURVATURE_X5_Y0", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X5_Y1": Attr("CURVATURE_X5_Y1", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X6_Y0": Attr("CURVATURE_X6_Y0", State.Free, Kind.Real, "1/m^5", ""),
}

by_element["E_GUN"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "DT_MAX": Attr("DT_MAX", State.Free, Kind.Real, "sec", ""),
    "GRADIENT": Attr("GRADIENT", State.Free, Kind.Real, "eV/m", "Gradient."),
    "VOLTAGE": Attr("VOLTAGE", State.Free, Kind.Real, "Volt", "Voltage. Dependent attribute."),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "RF_FREQUENCY": Attr("RF_FREQUENCY", State.Free, Kind.Real, "Hz", "Frequency of the RF field."),
    "RF_WAVELENGTH": Attr("RF_WAVELENGTH", State.Dependent, Kind.Real, "m", ""),
    "AUTOSCALE_AMPLITUDE": Attr("AUTOSCALE_AMPLITUDE", State.Free, Kind.Logical, "", ""),
    "AUTOSCALE_PHASE": Attr("AUTOSCALE_PHASE", State.Free, Kind.Logical, "", ""),
    "EMIT_FRACTION": Attr("EMIT_FRACTION", State.Free, Kind.Real, "", ""),
    "PHI0": Attr(
        "PHI0",
        State.Free,
        Kind.Real,
        "rad/2pi",
        "Phase (rad/2\\(\\pi\\)) of the reference particle with",
    ),
    "PHI0_MULTIPASS": Attr("PHI0_MULTIPASS", State.Private, Kind.Unknown, "", ""),
    "PHI0_AUTOSCALE": Attr("PHI0_AUTOSCALE", State.Quasi_Free, Kind.Real, "rad/2pi", ""),
    "FIELD_AUTOSCALE": Attr("FIELD_AUTOSCALE", State.Quasi_Free, Kind.Real, "", ""),
    "DELTA_E_REF": Attr("DELTA_E_REF", State.Free, Kind.Real, "eV", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "P0C_REF_INIT": Attr("P0C_REF_INIT", State.Private, Kind.Unknown, "", ""),
    "E_TOT_REF_INIT": Attr("E_TOT_REF_INIT", State.Private, Kind.Unknown, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["EM_FIELD"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "RF_FREQUENCY": Attr("RF_FREQUENCY", State.Free, Kind.Real, "Hz", ""),
    "RF_WAVELENGTH": Attr("RF_WAVELENGTH", State.Dependent, Kind.Real, "m", ""),
    "CONSTANT_REF_ENERGY": Attr(
        "CONSTANT_REF_ENERGY",
        State.Free,
        Kind.Logical,
        "",
        "Is the reference energy constant? Default = True.",
    ),
    "AUTOSCALE_AMPLITUDE": Attr("AUTOSCALE_AMPLITUDE", State.Free, Kind.Logical, "", ""),
    "AUTOSCALE_PHASE": Attr("AUTOSCALE_PHASE", State.Free, Kind.Logical, "", ""),
    "POLARITY": Attr("POLARITY", State.Free, Kind.Real, "", "For scaling the field."),
    "PHI0": Attr("PHI0", State.Free, Kind.Real, "rad/2pi", ""),
    "PHI0_ERR": Attr("PHI0_ERR", State.Free, Kind.Real, "rad/2pi", ""),
    "PHI0_AUTOSCALE": Attr("PHI0_AUTOSCALE", State.Quasi_Free, Kind.Real, "rad/2pi", ""),
    "FIELD_AUTOSCALE": Attr("FIELD_AUTOSCALE", State.Quasi_Free, Kind.Real, "", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Quasi_Free,
        Kind.Real,
        "eV",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Quasi_Free,
        Kind.Real,
        "eV",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Dependent, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Dependent, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["FLOOR_SHIFT"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length"),
    "TILT": Attr("TILT", State.Free, Kind.Real, "rad", "Rotation of the reference coords."),
    "ORIGIN_ELE_REF_PT": Attr(
        "ORIGIN_ELE_REF_PT", State.Free, Kind.Switch, "", "Reference pt on the reference ele."
    ),
    "UPSTREAM_ELE_DIR": Attr("UPSTREAM_ELE_DIR", State.Dependent, Kind.Integer, "", ""),
    "DOWNSTREAM_ELE_DIR": Attr("DOWNSTREAM_ELE_DIR", State.Dependent, Kind.Integer, "", ""),
    "X_PITCH": Attr("X_PITCH", State.Free, Kind.Real, "rad", "Rotation of the reference coords."),
    "Y_PITCH": Attr("Y_PITCH", State.Free, Kind.Real, "rad", "Rotation of the reference coords."),
    "X_OFFSET": Attr("X_OFFSET", State.Free, Kind.Real, "m", "X offset from origin point."),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Y offset from origin point."),
    "Z_OFFSET": Attr("Z_OFFSET", State.Free, Kind.Real, "m", "Z offset from origin point."),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "ORIGIN_ELE": Attr("ORIGIN_ELE", State.Free, Kind.String, "", "Reference element."),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["FIDUCIAL"] = {
    "L": Attr("L", State.Dependent, Kind.Real, "m", "Length of the element."),
    "ORIGIN_ELE_REF_PT": Attr(
        "ORIGIN_ELE_REF_PT", State.Free, Kind.Switch, "", "Reference pt on reference ele."
    ),
    "DX_ORIGIN": Attr("DX_ORIGIN", State.Free, Kind.Real, "m", "X-position offset"),
    "DY_ORIGIN": Attr("DY_ORIGIN", State.Free, Kind.Real, "m", "Y-position offset"),
    "DZ_ORIGIN": Attr("DZ_ORIGIN", State.Free, Kind.Real, "m", "Z-position offset"),
    "DTHETA_ORIGIN": Attr(
        "DTHETA_ORIGIN", State.Free, Kind.Real, "rad", "Orientation angle offset."
    ),
    "DPHI_ORIGIN": Attr("DPHI_ORIGIN", State.Free, Kind.Real, "rad", "Orientation angle offset."),
    "DPSI_ORIGIN": Attr("DPSI_ORIGIN", State.Free, Kind.Real, "rad", "Orientation angle offset."),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "ORIGIN_ELE": Attr("ORIGIN_ELE", State.Free, Kind.String, "", "Reference element."),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["UNDULATOR"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "KX": Attr("KX", State.Free, Kind.Real, "1/m", "Planar wiggler horizontal wave number."),
    "B_MAX": Attr(
        "B_MAX",
        State.Free,
        Kind.Real,
        "T",
        "Maximum magnetic field (in T) on the wiggler centerline.",
    ),
    "G_MAX": Attr(
        "G_MAX", State.Dependent, Kind.Real, "1/m", "Maximum bending strength. Dependent attribute."
    ),
    "OSC_AMPLITUDE": Attr(
        "OSC_AMPLITUDE",
        State.Dependent,
        Kind.Real,
        "m",
        "Amplitude of the particle oscillations. Dependent attribute.",
    ),
    "K1X": Attr(
        "K1X",
        State.Dependent,
        Kind.Real,
        "1/m^2",
        "Planar wiggler horizontal defocusing strength. Dep attribute.",
    ),
    "K1Y": Attr(
        "K1Y",
        State.Dependent,
        Kind.Real,
        "1/m^2",
        "Planar wiggler vertical focusing strength. Dep attribute.",
    ),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "R0_ELEC": Attr("R0_ELEC", State.Free, Kind.Real, "m", ""),
    "R0_MAG": Attr("R0_MAG", State.Free, Kind.Real, "m", ""),
    "POLARITY": Attr("POLARITY", State.Free, Kind.Real, "", "For scaling the field."),
    "N_PERIOD": Attr(
        "N_PERIOD", State.Free, Kind.Real, "", "The number of periods. Dependent attribute."
    ),
    "L_PERIOD": Attr(
        "L_PERIOD",
        State.Free,
        Kind.Real,
        "m",
        "Length over which field vector returns to the same orientation.",
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "DELTA_REF_TIME_USER_SET": Attr(
        "DELTA_REF_TIME_USER_SET",
        State.Free,
        Kind.Logical,
        "",
        "Delta_ref_time set in lattice file.",
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME", State.Free, Kind.Real, "sec", "Reference time to traverse the element."
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "TERM": Attr("TERM", State.Free, Kind.Struct, "", ""),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["DIFFRACTION_PLATE"] = {
    "L": Attr("L", State.Private, Kind.Unknown, "", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "FIELD_SCALE_FACTOR": Attr(
        "FIELD_SCALE_FACTOR", State.Free, Kind.Real, "", "Factor to scale the photon field"
    ),
    "MODE": Attr("MODE", State.Free, Kind.Switch, "", "Reflection or transmission"),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "REF_WAVELENGTH": Attr(
        "REF_WAVELENGTH",
        State.Dependent,
        Kind.Real,
        "m",
        "Reference wavelength. Dependent attribute.",
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "DISPLACEMENT": Attr("DISPLACEMENT", State.Free, Kind.Struct, "", ""),
    "SEGMENTED": Attr("SEGMENTED", State.Free, Kind.Struct, "", ""),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "CURVATURE": Attr("CURVATURE", State.Free, Kind.Struct, "", ""),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "CURVATURE_X0_Y2": Attr("CURVATURE_X0_Y2", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X0_Y3": Attr("CURVATURE_X0_Y3", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X0_Y4": Attr("CURVATURE_X0_Y4", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X0_Y5": Attr("CURVATURE_X0_Y5", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X0_Y6": Attr("CURVATURE_X0_Y6", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X1_Y1": Attr("CURVATURE_X1_Y1", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X1_Y2": Attr("CURVATURE_X1_Y2", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X1_Y3": Attr("CURVATURE_X1_Y3", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X1_Y4": Attr("CURVATURE_X1_Y4", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X1_Y5": Attr("CURVATURE_X1_Y5", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X2_Y0": Attr("CURVATURE_X2_Y0", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X2_Y1": Attr("CURVATURE_X2_Y1", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X2_Y2": Attr("CURVATURE_X2_Y2", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X2_Y3": Attr("CURVATURE_X2_Y3", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X2_Y4": Attr("CURVATURE_X2_Y4", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X3_Y0": Attr("CURVATURE_X3_Y0", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X3_Y1": Attr("CURVATURE_X3_Y1", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X3_Y2": Attr("CURVATURE_X3_Y2", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X3_Y3": Attr("CURVATURE_X3_Y3", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X4_Y0": Attr("CURVATURE_X4_Y0", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X4_Y1": Attr("CURVATURE_X4_Y1", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X4_Y2": Attr("CURVATURE_X4_Y2", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X5_Y0": Attr("CURVATURE_X5_Y0", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X5_Y1": Attr("CURVATURE_X5_Y1", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X6_Y0": Attr("CURVATURE_X6_Y0", State.Free, Kind.Real, "1/m^5", ""),
}

by_element["PHOTON_INIT"] = {
    "L": Attr("L", State.Dependent, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "SIG_X": Attr("SIG_X", State.Free, Kind.Real, "m", "For x distribution"),
    "SIG_Y": Attr("SIG_Y", State.Free, Kind.Real, "m", "For y distribution"),
    "SIG_Z": Attr("SIG_Z", State.Free, Kind.Real, "m", "For z distribution"),
    "SIG_VX": Attr("SIG_VX", State.Free, Kind.Real, "m/s", ""),
    "SIG_VY": Attr("SIG_VY", State.Free, Kind.Real, "m/s", ""),
    "SIG_E": Attr("SIG_E", State.Free, Kind.Real, "eV", ""),
    "SIG_E2": Attr("SIG_E2", State.Free, Kind.Real, "eV", ""),
    "E_CENTER": Attr("E_CENTER", State.Free, Kind.Real, "eV", ""),
    "E2_CENTER": Attr("E2_CENTER", State.Free, Kind.Real, "eV", ""),
    "E2_PROBABILITY": Attr("E2_PROBABILITY", State.Free, Kind.Real, "", ""),
    "E_CENTER_RELATIVE_TO_REF": Attr("E_CENTER_RELATIVE_TO_REF", State.Free, Kind.Logical, "", ""),
    "SPATIAL_DISTRIBUTION": Attr(
        "SPATIAL_DISTRIBUTION", State.Free, Kind.Switch, "", "Gaussian or uniform."
    ),
    "VELOCITY_DISTRIBUTION": Attr(
        "VELOCITY_DISTRIBUTION", State.Free, Kind.Switch, "", "Gaussian, spherical, or uniform."
    ),
    "ENERGY_DISTRIBUTION": Attr(
        "ENERGY_DISTRIBUTION", State.Free, Kind.Switch, "", "Gaussian, uniform, or curve."
    ),
    "E_FIELD_X": Attr(
        "E_FIELD_X", State.Free, Kind.Real, "V/m", "Polarization. x & y = 0 -> random"
    ),
    "E_FIELD_Y": Attr("E_FIELD_Y", State.Free, Kind.Real, "V/m", ""),
    "SCALE_FIELD_TO_ONE": Attr("SCALE_FIELD_TO_ONE", State.Free, Kind.Logical, "", ""),
    "TRANSVERSE_SIGMA_CUT": Attr("TRANSVERSE_SIGMA_CUT", State.Free, Kind.Real, "", ""),
    "DS_SLICE": Attr("DS_SLICE", State.Free, Kind.Real, "m", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "REF_WAVELENGTH": Attr(
        "REF_WAVELENGTH", State.Dependent, Kind.Real, "m", "Ref wavelength. Dep attribute."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "ENERGY_PROBABILITY_CURVE": Attr("ENERGY_PROBABILITY_CURVE", State.Free, Kind.Struct, "", ""),
    "PHYSICAL_SOURCE": Attr(
        "PHYSICAL_SOURCE", State.Free, Kind.String, "", "Physical source of x-rays"
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["SAMPLE"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "MODE": Attr("MODE", State.Free, Kind.Switch, "", "Reflection or transmission."),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "DISPLACEMENT": Attr("DISPLACEMENT", State.Free, Kind.Struct, "", ""),
    "SEGMENTED": Attr("SEGMENTED", State.Free, Kind.Struct, "", ""),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "CURVATURE": Attr("CURVATURE", State.Free, Kind.Struct, "", ""),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "MATERIAL_TYPE": Attr("MATERIAL_TYPE", State.Free, Kind.String, "", ""),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "CURVATURE_X0_Y2": Attr("CURVATURE_X0_Y2", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X0_Y3": Attr("CURVATURE_X0_Y3", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X0_Y4": Attr("CURVATURE_X0_Y4", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X0_Y5": Attr("CURVATURE_X0_Y5", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X0_Y6": Attr("CURVATURE_X0_Y6", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X1_Y1": Attr("CURVATURE_X1_Y1", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X1_Y2": Attr("CURVATURE_X1_Y2", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X1_Y3": Attr("CURVATURE_X1_Y3", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X1_Y4": Attr("CURVATURE_X1_Y4", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X1_Y5": Attr("CURVATURE_X1_Y5", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X2_Y0": Attr("CURVATURE_X2_Y0", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X2_Y1": Attr("CURVATURE_X2_Y1", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X2_Y2": Attr("CURVATURE_X2_Y2", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X2_Y3": Attr("CURVATURE_X2_Y3", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X2_Y4": Attr("CURVATURE_X2_Y4", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X3_Y0": Attr("CURVATURE_X3_Y0", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X3_Y1": Attr("CURVATURE_X3_Y1", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X3_Y2": Attr("CURVATURE_X3_Y2", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X3_Y3": Attr("CURVATURE_X3_Y3", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X4_Y0": Attr("CURVATURE_X4_Y0", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X4_Y1": Attr("CURVATURE_X4_Y1", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X4_Y2": Attr("CURVATURE_X4_Y2", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X5_Y0": Attr("CURVATURE_X5_Y0", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X5_Y1": Attr("CURVATURE_X5_Y1", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X6_Y0": Attr("CURVATURE_X6_Y0", State.Free, Kind.Real, "1/m^5", ""),
}

by_element["DETECTOR"] = {
    "L": Attr("L", State.Dependent, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "X_GAIN_ERR": Attr("X_GAIN_ERR", State.Free, Kind.Real, "m", ""),
    "Y_GAIN_ERR": Attr("Y_GAIN_ERR", State.Free, Kind.Real, "m", ""),
    "CRUNCH": Attr("CRUNCH", State.Free, Kind.Real, "rad", ""),
    "NOISE": Attr("NOISE", State.Free, Kind.Real, "", ""),
    "OSC_AMPLITUDE": Attr("OSC_AMPLITUDE", State.Free, Kind.Real, "m", ""),
    "X_GAIN_CALIB": Attr("X_GAIN_CALIB", State.Free, Kind.Real, "m", ""),
    "Y_GAIN_CALIB": Attr("Y_GAIN_CALIB", State.Free, Kind.Real, "m", ""),
    "CRUNCH_CALIB": Attr("CRUNCH_CALIB", State.Free, Kind.Real, "rad", ""),
    "X_OFFSET_CALIB": Attr("X_OFFSET_CALIB", State.Free, Kind.Real, "m", ""),
    "Y_OFFSET_CALIB": Attr("Y_OFFSET_CALIB", State.Free, Kind.Real, "m", ""),
    "TILT_CALIB": Attr("TILT_CALIB", State.Free, Kind.Real, "rad", ""),
    "DE_ETA_MEAS": Attr("DE_ETA_MEAS", State.Free, Kind.Real, "", ""),
    "N_SAMPLE": Attr("N_SAMPLE", State.Free, Kind.Real, "", ""),
    "X_DISPERSION_ERR": Attr("X_DISPERSION_ERR", State.Free, Kind.Real, "m", ""),
    "Y_DISPERSION_ERR": Attr("Y_DISPERSION_ERR", State.Free, Kind.Real, "m", ""),
    "X_DISPERSION_CALIB": Attr("X_DISPERSION_CALIB", State.Free, Kind.Real, "m", ""),
    "Y_DISPERSION_CALIB": Attr("Y_DISPERSION_CALIB", State.Free, Kind.Real, "m", ""),
    "SPLIT_ID": Attr("SPLIT_ID", State.Private, Kind.Unknown, "", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "PIXEL": Attr("PIXEL", State.Free, Kind.Struct, "", ""),
    "DISPLACEMENT": Attr("DISPLACEMENT", State.Free, Kind.Struct, "", ""),
    "SEGMENTED": Attr("SEGMENTED", State.Free, Kind.Struct, "", ""),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "CURVATURE": Attr("CURVATURE", State.Free, Kind.Struct, "", ""),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "CURVATURE_X0_Y2": Attr("CURVATURE_X0_Y2", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X0_Y3": Attr("CURVATURE_X0_Y3", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X0_Y4": Attr("CURVATURE_X0_Y4", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X0_Y5": Attr("CURVATURE_X0_Y5", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X0_Y6": Attr("CURVATURE_X0_Y6", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X1_Y1": Attr("CURVATURE_X1_Y1", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X1_Y2": Attr("CURVATURE_X1_Y2", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X1_Y3": Attr("CURVATURE_X1_Y3", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X1_Y4": Attr("CURVATURE_X1_Y4", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X1_Y5": Attr("CURVATURE_X1_Y5", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X2_Y0": Attr("CURVATURE_X2_Y0", State.Free, Kind.Real, "1/m", ""),
    "CURVATURE_X2_Y1": Attr("CURVATURE_X2_Y1", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X2_Y2": Attr("CURVATURE_X2_Y2", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X2_Y3": Attr("CURVATURE_X2_Y3", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X2_Y4": Attr("CURVATURE_X2_Y4", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X3_Y0": Attr("CURVATURE_X3_Y0", State.Free, Kind.Real, "1/m^2", ""),
    "CURVATURE_X3_Y1": Attr("CURVATURE_X3_Y1", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X3_Y2": Attr("CURVATURE_X3_Y2", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X3_Y3": Attr("CURVATURE_X3_Y3", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X4_Y0": Attr("CURVATURE_X4_Y0", State.Free, Kind.Real, "1/m^3", ""),
    "CURVATURE_X4_Y1": Attr("CURVATURE_X4_Y1", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X4_Y2": Attr("CURVATURE_X4_Y2", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X5_Y0": Attr("CURVATURE_X5_Y0", State.Free, Kind.Real, "1/m^4", ""),
    "CURVATURE_X5_Y1": Attr("CURVATURE_X5_Y1", State.Free, Kind.Real, "1/m^5", ""),
    "CURVATURE_X6_Y0": Attr("CURVATURE_X6_Y0", State.Free, Kind.Real, "1/m^5", ""),
}

by_element["SAD_MULT"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "RHO": Attr("RHO", State.Free, Kind.Real, "m", ""),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe. SAD equivalent: DISFRIN."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT", State.Free, Kind.Switch, "", "Where fringe is applied. SAD equivalent: FRINGE."
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "KS": Attr("KS", State.Free, Kind.Real, "1/m", "Solenoid strength."),
    "E1": Attr("E1", State.Free, Kind.Real, "rad", "Bend face angles."),
    "E2": Attr("E2", State.Free, Kind.Real, "rad", "Bend face angles."),
    "FB1": Attr(
        "FB1", State.Free, Kind.Real, "m", "Bend edge fringe parameters. SAD equivalents: FB1, FB2."
    ),
    "FB2": Attr(
        "FB2", State.Free, Kind.Real, "m", "Bend edge fringe parameters. SAD equivalents: FB1, FB2."
    ),
    "FQ1": Attr(
        "FQ1", State.Free, Kind.Real, "m", "Quadrupole fringe parameters. SAD equivalents: F1, F2."
    ),
    "FQ2": Attr(
        "FQ2", State.Free, Kind.Real, "m", "Quadrupole fringe parameters. SAD equivalents: F1, F2."
    ),
    "EPS_STEP_SCALE": Attr(
        "EPS_STEP_SCALE",
        State.Free,
        Kind.Real,
        "m",
        "Step size scale. Default = 1. SAD equivalent: EPS.",
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "X_OFFSET_MULT": Attr(
        "X_OFFSET_MULT", State.Free, Kind.Real, "m", "Mult component offset. SAD equivalent: DX."
    ),
    "Y_OFFSET_MULT": Attr(
        "Y_OFFSET_MULT", State.Free, Kind.Real, "m", "Mult component offset. SAD equivalent: DY."
    ),
    "BS_FIELD": Attr("BS_FIELD", State.Free, Kind.Real, "T", "Solenoid field. SAD equivalent: BZ."),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Dependent, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Dependent, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["MASK"] = {
    "L": Attr("L", State.Private, Kind.Unknown, "", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "FIELD_SCALE_FACTOR": Attr(
        "FIELD_SCALE_FACTOR", State.Free, Kind.Real, "", "Factor to scale the photon field."
    ),
    "MODE": Attr(
        "MODE", State.Free, Kind.Switch, "", "Reflection or transmission (photon tracking only)."
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "REF_WAVELENGTH": Attr(
        "REF_WAVELENGTH", State.Dependent, Kind.Real, "m", "Reference wavelength. Dependent attrib."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["AC_KICKER"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "INTERPOLATION": Attr(
        "INTERPOLATION", State.Free, Kind.Switch, "", "Cubic (default) or linear."
    ),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "R0_ELEC": Attr("R0_ELEC", State.Free, Kind.Real, "m", ""),
    "R0_MAG": Attr("R0_MAG", State.Free, Kind.Real, "m", ""),
    "PHI0_MULTIPASS": Attr("PHI0_MULTIPASS", State.Free, Kind.Real, "rad/2pi", ""),
    "T_OFFSET": Attr("T_OFFSET", State.Free, Kind.Real, "sec", "Time offset of field waveform."),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "FREQUENCIES": Attr("FREQUENCIES", State.Free, Kind.Struct, "", ""),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "AMP_VS_TIME": Attr("AMP_VS_TIME", State.Free, Kind.Struct, "", ""),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["LENS"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "RADIUS": Attr("RADIUS", State.Free, Kind.Real, "m", ""),
    "FOCAL_STRENGTH": Attr("FOCAL_STRENGTH", State.Free, Kind.Real, "1/m", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["CRAB_CAVITY"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "HARMON": Attr("HARMON", State.Free, Kind.Real, "", "Harmonic number"),
    "HARMON_MASTER": Attr(
        "HARMON_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "Is harmon or rf_frequency the dependent variable",
    ),
    "GRADIENT": Attr(
        "GRADIENT", State.Dependent, Kind.Real, "eV/m", "Accelerating gradient (V/m)."
    ),
    "VOLTAGE": Attr(
        "VOLTAGE", State.Free, Kind.Real, "Volt", "Cavity voltage. Dependent attribute."
    ),
    "RF_FREQUENCY": Attr("RF_FREQUENCY", State.Free, Kind.Real, "Hz", "RF frequency (Hz)."),
    "RF_WAVELENGTH": Attr("RF_WAVELENGTH", State.Dependent, Kind.Real, "m", ""),
    "AUTOSCALE_AMPLITUDE": Attr("AUTOSCALE_AMPLITUDE", State.Private, Kind.Logical, "", ""),
    "AUTOSCALE_PHASE": Attr("AUTOSCALE_PHASE", State.Private, Kind.Logical, "", ""),
    "PHI0": Attr(
        "PHI0",
        State.Free,
        Kind.Real,
        "rad/2pi",
        "Phase (rad/2\\(\\pi\\)) of the reference particle with",
    ),
    "PHI0_MULTIPASS": Attr(
        "PHI0_MULTIPASS",
        State.Free,
        Kind.Real,
        "rad/2pi",
        "Phase (rad/2\\(\\pi\\)) with respect to a multipass lord.",
    ),
    "PHI0_AUTOSCALE": Attr("PHI0_AUTOSCALE", State.Private, Kind.Unknown, "rad/2pi", ""),
    "FIELD_AUTOSCALE": Attr("FIELD_AUTOSCALE", State.Private, Kind.Unknown, "", ""),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "N_RF_STEPS": Attr("N_RF_STEPS", State.Free, Kind.Integer, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["RAMPER"] = {
    "INTERPOLATION": Attr("INTERPOLATION", State.Free, Kind.Switch, "", ""),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "X_KNOT": Attr("X_KNOT", State.Free, Kind.Struct, "", ""),
    "Y_KNOT": Attr("Y_KNOT", State.Free, Kind.Real, "", ""),
    "SLAVE": Attr("SLAVE", State.Free, Kind.Real, "", ""),
    "VAR": Attr("VAR", State.Free, Kind.Struct, "", ""),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
}

by_element["!PTC_COM"] = {
    "MAX_FRINGE_ORDER": Attr("MAX_FRINGE_ORDER", State.Free, Kind.Integer, "", ""),
    "EXACT_MISALIGN": Attr("EXACT_MISALIGN", State.Free, Kind.Logical, "", ""),
    "OLD_INTEGRATOR": Attr("OLD_INTEGRATOR", State.Free, Kind.Logical, "", ""),
    "EXACT_MODEL": Attr("EXACT_MODEL", State.Free, Kind.Logical, "", ""),
    "VERTICAL_KICK": Attr("VERTICAL_KICK", State.Free, Kind.Integer, "", ""),
}

by_element["RF_BEND"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", '"Length" of bend.'),
    "ROLL": Attr(
        "ROLL", State.Free, Kind.Real, "rad", "Rotation of the element about the longitudinal axis."
    ),
    "REF_TILT": Attr(
        "REF_TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element and reference orbit about the s axis.",
    ),
    "HARMON": Attr("HARMON", State.Quasi_Free, Kind.Real, "", "Harmonic number"),
    "HARMON_MASTER": Attr(
        "HARMON_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "Is harmon or rf_frequency the dependent var with ref energy changes?",
    ),
    "G": Attr("G", State.Quasi_Free, Kind.Real, "1/m", "Design bend strength (= 1/rho)."),
    "DG": Attr("DG", State.Private, Kind.Unknown, "1/m", ""),
    "RHO": Attr("RHO", State.Quasi_Free, Kind.Real, "m", "Design bend radius. Dependent param."),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "RF_FREQUENCY": Attr("RF_FREQUENCY", State.Quasi_Free, Kind.Real, "Hz", "Frequency"),
    "RF_WAVELENGTH": Attr("RF_WAVELENGTH", State.Dependent, Kind.Real, "m", ""),
    "PHI0": Attr("PHI0", State.Free, Kind.Real, "rad/2pi", "Cavity phase (rad/2pi)."),
    "PHI0_MULTIPASS": Attr(
        "PHI0_MULTIPASS",
        State.Free,
        Kind.Real,
        "rad/2pi",
        "Phase variation with multipass elements (rad/2pi).",
    ),
    "L_RECTANGLE": Attr("L_RECTANGLE", State.Quasi_Free, Kind.Real, "m", ""),
    "L_SAGITTA": Attr(
        "L_SAGITTA", State.Dependent, Kind.Real, "m", "Sagittal length. Dependent param."
    ),
    "L_CHORD": Attr("L_CHORD", State.Quasi_Free, Kind.Real, "m", "Chord length. See."),
    "FIDUCIAL_PT": Attr("FIDUCIAL_PT", State.Free, Kind.Switch, "", ""),
    "INIT_NEEDED": Attr("INIT_NEEDED", State.Private, Kind.Unknown, "", ""),
    "ANGLE": Attr("ANGLE", State.Quasi_Free, Kind.Real, "rad", "Design bend angle. Dependent var."),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "B_FIELD": Attr(
        "B_FIELD", State.Quasi_Free, Kind.Real, "T", "Design field strength (= P_0 g / q)."
    ),
    "DB_FIELD": Attr("DB_FIELD", State.Private, Kind.Unknown, "T", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "ROLL_TOT": Attr(
        "ROLL_TOT", State.Dependent, Kind.Real, "rad", "Net roll including support misalignments."
    ),
    "REF_TILT_TOT": Attr("REF_TILT_TOT", State.Dependent, Kind.Real, "rad", ""),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["GKICKER"] = {
    "X_KICK": Attr("X_KICK", State.Free, Kind.Real, "m", "X-position kick"),
    "PX_KICK": Attr("PX_KICK", State.Free, Kind.Real, "", "X-momentum kick"),
    "Y_KICK": Attr("Y_KICK", State.Free, Kind.Real, "m", "Y-position kick"),
    "PY_KICK": Attr("PY_KICK", State.Free, Kind.Real, "", "Y-momentum kick"),
    "Z_KICK": Attr("Z_KICK", State.Free, Kind.Real, "m", "Z-position kick"),
    "PZ_KICK": Attr("PZ_KICK", State.Free, Kind.Real, "", "Momentum kick"),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["FOIL"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "FINAL_CHARGE": Attr("FINAL_CHARGE", State.Free, Kind.Integer, "", "Final charge state"),
    "SCATTER_TEST": Attr(
        "SCATTER_TEST", State.Free, Kind.Logical, "", "For testing scattering. Default: False."
    ),
    "X1_EDGE": Attr(
        "X1_EDGE", State.Free, Kind.Real, "m", "Foil edge in the x-direction. Default: -99 m."
    ),
    "X2_EDGE": Attr(
        "X2_EDGE", State.Free, Kind.Real, "m", "Foil edge in the x-direction. Default: 99 m."
    ),
    "Y1_EDGE": Attr(
        "Y1_EDGE", State.Free, Kind.Real, "m", "Foil edge in the y-direction. Default: -99 m."
    ),
    "Y2_EDGE": Attr(
        "Y2_EDGE", State.Free, Kind.Real, "m", "Foil edge in the y-direction. Default: 99 m."
    ),
    "DTHICKNESS_DX": Attr(
        "DTHICKNESS_DX", State.Free, Kind.Real, "", "Wedge slope when the foil is wedge shaped."
    ),
    "F_FACTOR": Attr("F_FACTOR", State.Free, Kind.Real, "", ""),
    "SCATTER_METHOD": Attr(
        "SCATTER_METHOD", State.Free, Kind.Switch, "", "Scattering algorithm. Default: highland."
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "THICKNESS": Attr("THICKNESS", State.Free, Kind.Real, "m", "Material thickness (m)."),
    "NUM_STEPS": Attr("NUM_STEPS", State.Free, Kind.Integer, "", "Number of integration steps."),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "DENSITY": Attr("DENSITY", State.Free, Kind.Real, "kg/m^3", "Input material density (kg/m^3)."),
    "DENSITY_USED": Attr(
        "DENSITY_USED",
        State.Dependent,
        Kind.Real,
        "kg/m^3",
        "Density value used in tracking (kg/m^3).",
    ),
    "AREA_DENSITY": Attr(
        "AREA_DENSITY",
        State.Quasi_Free,
        Kind.Real,
        "kg/m^2",
        "Input material area density (kg/m^2).",
    ),
    "AREA_DENSITY_USED": Attr(
        "AREA_DENSITY_USED",
        State.Dependent,
        Kind.Real,
        "kg/m^2",
        "Area density used in tracking (kg/m^2).",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "RADIATION_LENGTH": Attr(
        "RADIATION_LENGTH", State.Free, Kind.Real, "kg/m^2", "Input material radiation length (m)."
    ),
    "RADIATION_LENGTH_USED": Attr(
        "RADIATION_LENGTH_USED",
        State.Dependent,
        Kind.Real,
        "kg/m^2",
        "Radiation length used in tracking (m).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "MATERIAL_TYPE": Attr("MATERIAL_TYPE", State.Free, Kind.String, "", "Foil material."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["THICK_MULTIPOLE"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "FRINGE_TYPE": Attr(
        "FRINGE_TYPE", State.Free, Kind.Switch, "", "Type of fringe field to apply."
    ),
    "FRINGE_AT": Attr(
        "FRINGE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Element end(s) at which fringe fields are applied.",
    ),
    "SPIN_FRINGE_ON": Attr(
        "SPIN_FRINGE_ON", State.Free, Kind.Logical, "", "If True, apply spin fringe-field kicks."
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "HKICK": Attr("HKICK", State.Quasi_Free, Kind.Real, "", "Horizontal kick."),
    "VKICK": Attr("VKICK", State.Quasi_Free, Kind.Real, "", "Vertical kick."),
    "BL_HKICK": Attr(
        "BL_HKICK", State.Quasi_Free, Kind.Real, "T*m", "Horizontal integrated field kick (B*L)."
    ),
    "BL_VKICK": Attr(
        "BL_VKICK", State.Quasi_Free, Kind.Real, "T*m", "Vertical integrated field kick (B*L)."
    ),
    "PTC_CANONICAL_COORDS": Attr("PTC_CANONICAL_COORDS", State.Free, Kind.Logical, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "MULTIPASS_REF_ENERGY": Attr("MULTIPASS_REF_ENERGY", State.Private, Kind.Switch, "", ""),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "INTEGRATOR_ORDER": Attr(
        "INTEGRATOR_ORDER", State.Free, Kind.Integer, "", "Order of the symplectic integrator."
    ),
    "NUM_STEPS": Attr(
        "NUM_STEPS", State.Quasi_Free, Kind.Integer, "", "Number of integration steps."
    ),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "CSR_DS_STEP": Attr(
        "CSR_DS_STEP", State.Free, Kind.Real, "m", "Integration step length for CSR / space-charge."
    ),
    "LORD_PAD1": Attr("LORD_PAD1", State.Quasi_Free, Kind.Real, "m", ""),
    "LORD_PAD2": Attr("LORD_PAD2", State.Quasi_Free, Kind.Real, "m", ""),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "LR_SELF_WAKE_ON": Attr(
        "LR_SELF_WAKE_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the long-range self-wake.",
    ),
    "LR_WAKE_FILE": Attr(
        "LR_WAKE_FILE", State.Free, Kind.String, "", "File defining the long-range wakefield."
    ),
    "LR_FREQ_SPREAD": Attr(
        "LR_FREQ_SPREAD",
        State.Free,
        Kind.Real,
        "Hz",
        "Fractional spread in long-range wake mode frequencies.",
    ),
    "MULTIPOLES_ON": Attr(
        "MULTIPOLES_ON",
        State.Free,
        Kind.Logical,
        "",
        "If True, include the element's multipole fields.",
    ),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "TAYLOR_MAP_INCLUDES_OFFSETS": Attr(
        "TAYLOR_MAP_INCLUDES_OFFSETS",
        State.Free,
        Kind.Logical,
        "",
        "If True, the Taylor map folds in the element's offsets, pitches, and tilt.",
    ),
    "CSR_METHOD": Attr(
        "CSR_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Coherent synchrotron radiation calculation method.",
    ),
    "SPACE_CHARGE_METHOD": Attr(
        "SPACE_CHARGE_METHOD", State.Free, Kind.Switch, "", "Space charge calculation method."
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "SR_WAKE_FILE": Attr(
        "SR_WAKE_FILE", State.Free, Kind.String, "", "File defining the short-range wakefield."
    ),
    "SYMPLECTIFY": Attr(
        "SYMPLECTIFY",
        State.Free,
        Kind.Logical,
        "",
        "If True, make the transfer map exactly symplectic.",
    ),
    "FIELD_CALC": Attr(
        "FIELD_CALC",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the electromagnetic field.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "FIELD_OVERLAPS": Attr(
        "FIELD_OVERLAPS", State.Free, Kind.Struct, "", "Elements whose fields overlap this one."
    ),
    "FIELD_MASTER": Attr(
        "FIELD_MASTER",
        State.Free,
        Kind.Logical,
        "",
        "If True, unnormalized field strengths are the independent parameters; if False, normalized strengths are.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "SCALE_MULTIPOLES": Attr(
        "SCALE_MULTIPOLES",
        State.Free,
        Kind.Logical,
        "",
        "If True, scale multipoles by the element's strength.",
    ),
    "SR_WAKE": Attr("SR_WAKE", State.Free, Kind.Struct, "", "Short-range wakefield definition."),
    "LR_WAKE": Attr("LR_WAKE", State.Free, Kind.Struct, "", "Long-range wakefield definition."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CARTESIAN_MAP": Attr(
        "CARTESIAN_MAP", State.Free, Kind.Struct, "", "Field map defined in Cartesian coordinates."
    ),
    "CYLINDRICAL_MAP": Attr(
        "CYLINDRICAL_MAP",
        State.Free,
        Kind.Struct,
        "",
        "Field map defined in cylindrical coordinates.",
    ),
    "GRID_FIELD": Attr(
        "GRID_FIELD", State.Free, Kind.Struct, "", "Field defined on a grid of points."
    ),
    "GEN_GRADIENTS": Attr(
        "GEN_GRADIENTS", State.Free, Kind.Struct, "", "Field defined by generalized gradients."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
    "A0": Attr("A0", State.Free, Kind.Real, "", "Skew magnetic multipole coefficient of order 0."),
    "A1": Attr(
        "A1", State.Free, Kind.Real, "1/m", "Skew magnetic multipole coefficient of order 1."
    ),
    "A2": Attr(
        "A2", State.Free, Kind.Real, "1/m^2", "Skew magnetic multipole coefficient of order 2."
    ),
    "A3": Attr(
        "A3", State.Free, Kind.Real, "1/m^3", "Skew magnetic multipole coefficient of order 3."
    ),
    "A4": Attr(
        "A4", State.Free, Kind.Real, "1/m^4", "Skew magnetic multipole coefficient of order 4."
    ),
    "A5": Attr(
        "A5", State.Free, Kind.Real, "1/m^5", "Skew magnetic multipole coefficient of order 5."
    ),
    "A6": Attr(
        "A6", State.Free, Kind.Real, "1/m^6", "Skew magnetic multipole coefficient of order 6."
    ),
    "A7": Attr(
        "A7", State.Free, Kind.Real, "1/m^7", "Skew magnetic multipole coefficient of order 7."
    ),
    "A8": Attr(
        "A8", State.Free, Kind.Real, "1/m^8", "Skew magnetic multipole coefficient of order 8."
    ),
    "A9": Attr(
        "A9", State.Free, Kind.Real, "1/m^9", "Skew magnetic multipole coefficient of order 9."
    ),
    "A10": Attr(
        "A10", State.Free, Kind.Real, "1/m^10", "Skew magnetic multipole coefficient of order 10."
    ),
    "A11": Attr(
        "A11", State.Free, Kind.Real, "1/m^11", "Skew magnetic multipole coefficient of order 11."
    ),
    "A12": Attr(
        "A12", State.Free, Kind.Real, "1/m^12", "Skew magnetic multipole coefficient of order 12."
    ),
    "A13": Attr(
        "A13", State.Free, Kind.Real, "1/m^13", "Skew magnetic multipole coefficient of order 13."
    ),
    "A14": Attr(
        "A14", State.Free, Kind.Real, "1/m^14", "Skew magnetic multipole coefficient of order 14."
    ),
    "A15": Attr(
        "A15", State.Free, Kind.Real, "1/m^15", "Skew magnetic multipole coefficient of order 15."
    ),
    "A16": Attr(
        "A16", State.Free, Kind.Real, "1/m^16", "Skew magnetic multipole coefficient of order 16."
    ),
    "A17": Attr(
        "A17", State.Free, Kind.Real, "1/m^17", "Skew magnetic multipole coefficient of order 17."
    ),
    "A18": Attr(
        "A18", State.Free, Kind.Real, "1/m^18", "Skew magnetic multipole coefficient of order 18."
    ),
    "A19": Attr(
        "A19", State.Free, Kind.Real, "1/m^19", "Skew magnetic multipole coefficient of order 19."
    ),
    "A20": Attr(
        "A20", State.Free, Kind.Real, "1/m^20", "Skew magnetic multipole coefficient of order 20."
    ),
    "A21": Attr(
        "A21", State.Free, Kind.Real, "1/m^21", "Skew magnetic multipole coefficient of order 21."
    ),
    "B0": Attr(
        "B0", State.Free, Kind.Real, "", "Normal magnetic multipole coefficient of order 0."
    ),
    "B1": Attr(
        "B1", State.Free, Kind.Real, "1/m", "Normal magnetic multipole coefficient of order 1."
    ),
    "B2": Attr(
        "B2", State.Free, Kind.Real, "1/m^2", "Normal magnetic multipole coefficient of order 2."
    ),
    "B3": Attr(
        "B3", State.Free, Kind.Real, "1/m^3", "Normal magnetic multipole coefficient of order 3."
    ),
    "B4": Attr(
        "B4", State.Free, Kind.Real, "1/m^4", "Normal magnetic multipole coefficient of order 4."
    ),
    "B5": Attr(
        "B5", State.Free, Kind.Real, "1/m^5", "Normal magnetic multipole coefficient of order 5."
    ),
    "B6": Attr(
        "B6", State.Free, Kind.Real, "1/m^6", "Normal magnetic multipole coefficient of order 6."
    ),
    "B7": Attr(
        "B7", State.Free, Kind.Real, "1/m^7", "Normal magnetic multipole coefficient of order 7."
    ),
    "B8": Attr(
        "B8", State.Free, Kind.Real, "1/m^8", "Normal magnetic multipole coefficient of order 8."
    ),
    "B9": Attr(
        "B9", State.Free, Kind.Real, "1/m^9", "Normal magnetic multipole coefficient of order 9."
    ),
    "B10": Attr(
        "B10", State.Free, Kind.Real, "1/m^10", "Normal magnetic multipole coefficient of order 10."
    ),
    "B11": Attr(
        "B11", State.Free, Kind.Real, "1/m^11", "Normal magnetic multipole coefficient of order 11."
    ),
    "B12": Attr(
        "B12", State.Free, Kind.Real, "1/m^12", "Normal magnetic multipole coefficient of order 12."
    ),
    "B13": Attr(
        "B13", State.Free, Kind.Real, "1/m^13", "Normal magnetic multipole coefficient of order 13."
    ),
    "B14": Attr(
        "B14", State.Free, Kind.Real, "1/m^14", "Normal magnetic multipole coefficient of order 14."
    ),
    "B15": Attr(
        "B15", State.Free, Kind.Real, "1/m^15", "Normal magnetic multipole coefficient of order 15."
    ),
    "B16": Attr(
        "B16", State.Free, Kind.Real, "1/m^16", "Normal magnetic multipole coefficient of order 16."
    ),
    "B17": Attr(
        "B17", State.Free, Kind.Real, "1/m^17", "Normal magnetic multipole coefficient of order 17."
    ),
    "B18": Attr(
        "B18", State.Free, Kind.Real, "1/m^18", "Normal magnetic multipole coefficient of order 18."
    ),
    "B19": Attr(
        "B19", State.Free, Kind.Real, "1/m^19", "Normal magnetic multipole coefficient of order 19."
    ),
    "B20": Attr(
        "B20", State.Free, Kind.Real, "1/m^20", "Normal magnetic multipole coefficient of order 20."
    ),
    "B21": Attr(
        "B21", State.Free, Kind.Real, "1/m^21", "Normal magnetic multipole coefficient of order 21."
    ),
}

by_element["PICKUP"] = {
    "L": Attr("L", State.Free, Kind.Real, "m", "Length of the element."),
    "TILT": Attr(
        "TILT",
        State.Free,
        Kind.Real,
        "rad",
        "Rotation of the element about the longitudinal (s) axis.",
    ),
    "X_PITCH": Attr(
        "X_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the x axis."
    ),
    "Y_PITCH": Attr(
        "Y_PITCH", State.Free, Kind.Real, "rad", "Pitch (rotation) of the element about the y axis."
    ),
    "X_OFFSET": Attr(
        "X_OFFSET", State.Free, Kind.Real, "m", "Horizontal (x) offset of the element."
    ),
    "Y_OFFSET": Attr("Y_OFFSET", State.Free, Kind.Real, "m", "Vertical (y) offset of the element."),
    "Z_OFFSET": Attr(
        "Z_OFFSET", State.Free, Kind.Real, "m", "Longitudinal (s) offset of the element."
    ),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "X_PITCH_TOT": Attr(
        "X_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net x pitch including support misalignments.",
    ),
    "Y_PITCH_TOT": Attr(
        "Y_PITCH_TOT",
        State.Dependent,
        Kind.Real,
        "rad",
        "Net y pitch including support misalignments.",
    ),
    "X_OFFSET_TOT": Attr(
        "X_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net horizontal offset including support misalignments.",
    ),
    "Y_OFFSET_TOT": Attr(
        "Y_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net vertical offset including support misalignments.",
    ),
    "Z_OFFSET_TOT": Attr(
        "Z_OFFSET_TOT",
        State.Dependent,
        Kind.Real,
        "m",
        "Net longitudinal offset including support misalignments.",
    ),
    "TILT_TOT": Attr(
        "TILT_TOT", State.Dependent, Kind.Real, "rad", "Net tilt including support misalignments."
    ),
    "DISPATCH": Attr("DISPATCH", State.Private, Kind.Unknown, "", ""),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "NUM_STEPS": Attr("NUM_STEPS", State.Free, Kind.Integer, "", "Number of integration steps."),
    "DS_STEP": Attr("DS_STEP", State.Free, Kind.Real, "m", "Length of an integration step."),
    "X1_LIMIT": Attr("X1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -x side."),
    "X2_LIMIT": Attr("X2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +x side."),
    "Y1_LIMIT": Attr("Y1_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the -y side."),
    "Y2_LIMIT": Attr("Y2_LIMIT", State.Free, Kind.Real, "m", "Aperture limit on the +y side."),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "APERTURE_TYPE": Attr(
        "APERTURE_TYPE",
        State.Free,
        Kind.Switch,
        "",
        "Aperture shape (rectangular, elliptical, ...).",
    ),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "APERTURE": Attr(
        "APERTURE", State.Free, Kind.Real, "m", "Aperture half-size; sets both x and y limits."
    ),
    "X_LIMIT": Attr("X_LIMIT", State.Free, Kind.Real, "m", "Horizontal aperture half-size."),
    "Y_LIMIT": Attr("Y_LIMIT", State.Free, Kind.Real, "m", "Vertical aperture half-size."),
    "OFFSET_MOVES_APERTURE": Attr(
        "OFFSET_MOVES_APERTURE",
        State.Free,
        Kind.Logical,
        "",
        "If True, element offsets shift the aperture too.",
    ),
    "WALL": Attr(
        "WALL", State.Free, Kind.Struct, "", "Vacuum chamber wall / aperture cross-section."
    ),
    "APERTURE_AT": Attr(
        "APERTURE_AT",
        State.Free,
        Kind.Switch,
        "",
        "Longitudinal location(s) where the aperture is applied.",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "CREATE_JUMBO_SLAVE": Attr(
        "CREATE_JUMBO_SLAVE",
        State.Free,
        Kind.Logical,
        "",
        "If True, create a single jumbo super-slave.",
    ),
    "ACCORDION_EDGE": Attr("ACCORDION_EDGE", State.Private, Kind.Unknown, "", ""),
    "START_EDGE": Attr("START_EDGE", State.Private, Kind.Unknown, "", ""),
    "END_EDGE": Attr("END_EDGE", State.Private, Kind.Unknown, "", ""),
    "S_POSITION": Attr("S_POSITION", State.Private, Kind.Unknown, "", ""),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}

by_element["FEEDBACK"] = {
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "INPUT_ELE": Attr(
        "INPUT_ELE",
        State.Free,
        Kind.String,
        "",
        "Lattice element(s) feedback element gets information from",
    ),
    "OUTPUT_ELE": Attr(
        "OUTPUT_ELE",
        State.Free,
        Kind.String,
        "",
        "Lattice elements(s) where the feedback element can influence",
    ),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
}

by_element["FIXER"] = {
    "IX_FIXER": Attr("IX_FIXER", State.Private, Kind.Unknown, "", ""),
    "SPIN_DN_DPZ_X": Attr("SPIN_DN_DPZ_X", State.Quasi_Free, Kind.Real, "", ""),
    "SPIN_DN_DPZ_Y": Attr("SPIN_DN_DPZ_Y", State.Quasi_Free, Kind.Real, "", ""),
    "SPIN_DN_DPZ_Z": Attr("SPIN_DN_DPZ_Z", State.Quasi_Free, Kind.Real, "", ""),
    "X_STORED": Attr("X_STORED", State.Free, Kind.Real, "m", ""),
    "PX_STORED": Attr("PX_STORED", State.Free, Kind.Real, "", ""),
    "Y_STORED": Attr("Y_STORED", State.Free, Kind.Real, "m", "Particle phase space coordinates"),
    "PY_STORED": Attr("PY_STORED", State.Free, Kind.Real, "", "Particle phase space coordinates"),
    "Z_STORED": Attr("Z_STORED", State.Free, Kind.Real, "m", ""),
    "PZ_STORED": Attr("PZ_STORED", State.Free, Kind.Real, "", ""),
    "BETA_A_STORED": Attr("BETA_A_STORED", State.Free, Kind.Real, "m", "a-mode Twiss"),
    "ALPHA_A_STORED": Attr("ALPHA_A_STORED", State.Free, Kind.Real, "", "a-mode Twiss"),
    "BETA_B_STORED": Attr("BETA_B_STORED", State.Free, Kind.Real, "m", "b-mode Twiss"),
    "ALPHA_B_STORED": Attr("ALPHA_B_STORED", State.Free, Kind.Real, "", "b-mode Twiss"),
    "PHI_A_STORED": Attr("PHI_A_STORED", State.Free, Kind.Real, "", "a-mode Twiss"),
    "PHI_B_STORED": Attr("PHI_B_STORED", State.Free, Kind.Real, "", "b-mode Twiss"),
    "MODE_FLIP_STORED": Attr(
        "MODE_FLIP_STORED", State.Free, Kind.Logical, "", "Logical: Normal modes flipped?"
    ),
    "CMAT_11": Attr("CMAT_11", State.Quasi_Free, Kind.Real, "", ""),
    "CMAT_12": Attr("CMAT_12", State.Quasi_Free, Kind.Real, "m", ""),
    "CMAT_21": Attr("CMAT_21", State.Quasi_Free, Kind.Real, "1/m", ""),
    "CMAT_22": Attr("CMAT_22", State.Quasi_Free, Kind.Real, "", ""),
    "MODE_FLIP": Attr("MODE_FLIP", State.Quasi_Free, Kind.Logical, "", ""),
    "ETA_X_STORED": Attr("ETA_X_STORED", State.Free, Kind.Real, "m", "Horizontal dispersion"),
    "ETAP_X_STORED": Attr("ETAP_X_STORED", State.Free, Kind.Real, "", "Horizontal dispersion"),
    "ETA_Y_STORED": Attr("ETA_Y_STORED", State.Free, Kind.Real, "m", "Vertical dispersion"),
    "ETAP_Y_STORED": Attr("ETAP_Y_STORED", State.Free, Kind.Real, "", "Vertical dispersion"),
    "CMAT_11_STORED": Attr("CMAT_11_STORED", State.Free, Kind.Real, "", "Coupling"),
    "CMAT_12_STORED": Attr("CMAT_12_STORED", State.Free, Kind.Real, "m", "Coupling"),
    "CMAT_21_STORED": Attr("CMAT_21_STORED", State.Free, Kind.Real, "1/m", ""),
    "CMAT_22_STORED": Attr("CMAT_22_STORED", State.Free, Kind.Real, "", ""),
    "DBETA_DPZ_A_STORED": Attr("DBETA_DPZ_A_STORED", State.Free, Kind.Real, "m", ""),
    "DBETA_DPZ_B_STORED": Attr("DBETA_DPZ_B_STORED", State.Free, Kind.Real, "m", ""),
    "DALPHA_DPZ_A_STORED": Attr("DALPHA_DPZ_A_STORED", State.Free, Kind.Real, "", ""),
    "DALPHA_DPZ_B_STORED": Attr("DALPHA_DPZ_B_STORED", State.Free, Kind.Real, "", ""),
    "DETA_DPZ_X_STORED": Attr("DETA_DPZ_X_STORED", State.Free, Kind.Real, "m", ""),
    "DETA_DPZ_Y_STORED": Attr("DETA_DPZ_Y_STORED", State.Free, Kind.Real, "m", ""),
    "DETAP_DPZ_X_STORED": Attr("DETAP_DPZ_X_STORED", State.Free, Kind.Real, "", ""),
    "DETAP_DPZ_Y_STORED": Attr("DETAP_DPZ_Y_STORED", State.Free, Kind.Real, "", ""),
    "DELTA_REF_TIME": Attr(
        "DELTA_REF_TIME",
        State.Dependent,
        Kind.Real,
        "sec",
        "Reference time to traverse the element.",
    ),
    "P0C_START": Attr(
        "P0C_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference momentum times c at the entrance end [eV].",
    ),
    "E_TOT_START": Attr(
        "E_TOT_START",
        State.Private,
        Kind.Unknown,
        "",
        "Reference total energy at the entrance end [eV].",
    ),
    "P0C": Attr(
        "P0C", State.Quasi_Free, Kind.Real, "eV", "Reference momentum times c at the exit end [eV]."
    ),
    "E_TOT": Attr(
        "E_TOT", State.Quasi_Free, Kind.Real, "eV", "Reference total energy at the exit end [eV]."
    ),
    "SPIN_X_STORED": Attr("SPIN_X_STORED", State.Free, Kind.Real, "", "Particle spin"),
    "SPIN_Y_STORED": Attr("SPIN_Y_STORED", State.Free, Kind.Real, "", "Particle spin"),
    "SPIN_Z_STORED": Attr("SPIN_Z_STORED", State.Free, Kind.Real, "", "Particle spin"),
    "REF_TIME_START": Attr(
        "REF_TIME_START", State.Dependent, Kind.Real, "sec", "Reference time at the entrance end."
    ),
    "DCMAT_DPZ_11_STORED": Attr("DCMAT_DPZ_11_STORED", State.Free, Kind.Real, "", ""),
    "DCMAT_DPZ_12_STORED": Attr("DCMAT_DPZ_12_STORED", State.Free, Kind.Real, "", ""),
    "DCMAT_DPZ_21_STORED": Attr("DCMAT_DPZ_21_STORED", State.Free, Kind.Real, "", ""),
    "DCMAT_DPZ_22_STORED": Attr("DCMAT_DPZ_22_STORED", State.Free, Kind.Real, "", ""),
    "CHECK_SUM": Attr("CHECK_SUM", State.Private, Kind.Unknown, "", ""),
    "IS_ON": Attr(
        "IS_ON",
        State.Free,
        Kind.Logical,
        "",
        "If False, the element's fields are turned off for tracking.",
    ),
    "ALIAS": Attr("ALIAS", State.Free, Kind.String, "", "User-defined alias name."),
    "ETA_X": Attr("ETA_X", State.Quasi_Free, Kind.Real, "m", ""),
    "ETA_Y": Attr("ETA_Y", State.Quasi_Free, Kind.Real, "m", ""),
    "ETAP_X": Attr("ETAP_X", State.Quasi_Free, Kind.Real, "", ""),
    "ETAP_Y": Attr("ETAP_Y", State.Quasi_Free, Kind.Real, "", ""),
    "PHI_A": Attr("PHI_A", State.Quasi_Free, Kind.Real, "rad", ""),
    "ETA_Z": Attr("ETA_Z", State.Quasi_Free, Kind.Real, "m", ""),
    "DETA_DPZ_X": Attr("DETA_DPZ_X", State.Quasi_Free, Kind.Real, "m", ""),
    "DETAP_DPZ_X": Attr("DETAP_DPZ_X", State.Quasi_Free, Kind.Real, "", ""),
    "MAT6_CALC_METHOD": Attr(
        "MAT6_CALC_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to compute the linear transfer matrix.",
    ),
    "TRACKING_METHOD": Attr(
        "TRACKING_METHOD",
        State.Free,
        Kind.Switch,
        "",
        "Method used to track particles through the element.",
    ),
    "PTC_INTEGRATION_TYPE": Attr(
        "PTC_INTEGRATION_TYPE", State.Free, Kind.Switch, "", "PTC integration type."
    ),
    "SPIN_TRACKING_METHOD": Attr(
        "SPIN_TRACKING_METHOD", State.Free, Kind.Switch, "", "Method used to track particle spin."
    ),
    "DETA_DPZ_Y": Attr("DETA_DPZ_Y", State.Quasi_Free, Kind.Real, "m", ""),
    "DETAP_DPZ_Y": Attr("DETAP_DPZ_Y", State.Quasi_Free, Kind.Real, "", ""),
    "ALPHA_A": Attr("ALPHA_A", State.Quasi_Free, Kind.Real, "", ""),
    "ALPHA_B": Attr("ALPHA_B", State.Quasi_Free, Kind.Real, "", ""),
    "S": Attr("S", State.Quasi_Free, Kind.Real, "m", ""),
    "X_POSITION": Attr("X_POSITION", State.Quasi_Free, Kind.Real, "m", ""),
    "Y_POSITION": Attr("Y_POSITION", State.Quasi_Free, Kind.Real, "m", ""),
    "Z_POSITION": Attr("Z_POSITION", State.Quasi_Free, Kind.Real, "m", ""),
    "THETA_POSITION": Attr("THETA_POSITION", State.Quasi_Free, Kind.Real, "rad", ""),
    "PHI_POSITION": Attr("PHI_POSITION", State.Quasi_Free, Kind.Real, "rad", ""),
    "PSI_POSITION": Attr("PSI_POSITION", State.Quasi_Free, Kind.Real, "rad", ""),
    "BETA_A": Attr("BETA_A", State.Quasi_Free, Kind.Real, "m", ""),
    "BETA_B": Attr("BETA_B", State.Quasi_Free, Kind.Real, "m", ""),
    "DBETA_DPZ_A": Attr("DBETA_DPZ_A", State.Quasi_Free, Kind.Real, "m", ""),
    "DBETA_DPZ_B": Attr("DBETA_DPZ_B", State.Quasi_Free, Kind.Real, "m", ""),
    "DESCRIP": Attr("DESCRIP", State.Free, Kind.String, "", "User-defined description string."),
    "DALPHA_DPZ_A": Attr("DALPHA_DPZ_A", State.Quasi_Free, Kind.Real, "", ""),
    "DALPHA_DPZ_B": Attr("DALPHA_DPZ_B", State.Quasi_Free, Kind.Real, "", ""),
    "PHI_B": Attr("PHI_B", State.Quasi_Free, Kind.Real, "rad", ""),
    "TYPE": Attr("TYPE", State.Free, Kind.String, "", "User-defined type string."),
    "REF_ORIGIN": Attr(
        "REF_ORIGIN",
        State.Free,
        Kind.Switch,
        "",
        "Reference-element origin point for superposition.",
    ),
    "ELE_ORIGIN": Attr(
        "ELE_ORIGIN", State.Free, Kind.Switch, "", "This element's origin point for superposition."
    ),
    "SUPERIMPOSE": Attr(
        "SUPERIMPOSE",
        State.Free,
        Kind.Struct,
        "",
        "If True, superimpose this element onto the lattice.",
    ),
    "OFFSET": Attr(
        "OFFSET",
        State.Free,
        Kind.Real,
        "m",
        "Longitudinal offset of the superposition reference point.",
    ),
    "REFERENCE": Attr(
        "REFERENCE", State.Free, Kind.String, "", "Element used as the superposition reference."
    ),
    "WRAP_SUPERIMPOSE": Attr(
        "WRAP_SUPERIMPOSE",
        State.Free,
        Kind.Logical,
        "",
        "If True, allow superposition to wrap around the ring.",
    ),
}
