"""
Curated descriptions for attributes common to many element types.

The manual (``elements.tex``) documents most attributes only in the sections of
the elements they are most associated with. Generic "housekeeping" attributes
(tracking methods, apertures, offsets, multipoles, ...) appear on dozens of
elements but are rarely re-documented per element. This hand-maintained table
fills those blanks.

Precedence in the generator: an element's own manual description wins; this
table is only consulted for attributes that section does not document. Keep
entries short and element-agnostic. Names are upper-cased.
"""

from __future__ import annotations

COMMON: dict[str, str] = {
    # Geometry / length
    "L": "Length of the element.",
    "TILT": "Rotation of the element about the longitudinal (s) axis.",
    "REF_TILT": "Rotation of the element and reference orbit about the s axis.",
    # Offsets and pitches (misalignments)
    "X_OFFSET": "Horizontal (x) offset of the element.",
    "Y_OFFSET": "Vertical (y) offset of the element.",
    "Z_OFFSET": "Longitudinal (s) offset of the element.",
    "X_PITCH": "Pitch (rotation) of the element about the x axis.",
    "Y_PITCH": "Pitch (rotation) of the element about the y axis.",
    "ROLL": "Rotation of the element about the longitudinal axis.",
    # Totals: value including support (girder / multipass) misalignments
    "X_OFFSET_TOT": "Net horizontal offset including support misalignments.",
    "Y_OFFSET_TOT": "Net vertical offset including support misalignments.",
    "Z_OFFSET_TOT": "Net longitudinal offset including support misalignments.",
    "X_PITCH_TOT": "Net x pitch including support misalignments.",
    "Y_PITCH_TOT": "Net y pitch including support misalignments.",
    "TILT_TOT": "Net tilt including support misalignments.",
    "ROLL_TOT": "Net roll including support misalignments.",
    # Reference energy / time
    "E_TOT": "Reference total energy at the exit end [eV].",
    "P0C": "Reference momentum times c at the exit end [eV].",
    "E_TOT_START": "Reference total energy at the entrance end [eV].",
    "P0C_START": "Reference momentum times c at the entrance end [eV].",
    "DELTA_REF_TIME": "Reference time to traverse the element.",
    "REF_TIME_START": "Reference time at the entrance end.",
    # On/off and field bookkeeping
    "IS_ON": "If False, the element's fields are turned off for tracking.",
    "FIELD_MASTER": "If True, unnormalized field strengths are the "
    "independent parameters; if False, normalized strengths are.",
    "MULTIPOLES_ON": "If True, include the element's multipole fields.",
    "SCALE_MULTIPOLES": "If True, scale multipoles by the element's strength.",
    "FIELD_CALC": "Method used to compute the electromagnetic field.",
    # Kicks
    "KICK": "Kick strength.",
    "HKICK": "Horizontal kick.",
    "VKICK": "Vertical kick.",
    "BL_HKICK": "Horizontal integrated field kick (B*L).",
    "BL_VKICK": "Vertical integrated field kick (B*L).",
    # Integration / tracking settings
    "DS_STEP": "Length of an integration step.",
    "CSR_DS_STEP": "Integration step length for CSR / space-charge.",
    "NUM_STEPS": "Number of integration steps.",
    "INTEGRATOR_ORDER": "Order of the symplectic integrator.",
    "TRACKING_METHOD": "Method used to track particles through the element.",
    "MAT6_CALC_METHOD": "Method used to compute the linear transfer matrix.",
    "SPIN_TRACKING_METHOD": "Method used to track particle spin.",
    "PTC_INTEGRATION_TYPE": "PTC integration type.",
    "CSR_METHOD": "Coherent synchrotron radiation calculation method.",
    "SPACE_CHARGE_METHOD": "Space charge calculation method.",
    "SYMPLECTIFY": "If True, make the transfer map exactly symplectic.",
    "TAYLOR_MAP_INCLUDES_OFFSETS": "If True, the Taylor map folds in the "
    "element's offsets, pitches, and tilt.",
    "SPIN_FRINGE_ON": "If True, apply spin fringe-field kicks.",
    # Apertures
    "APERTURE": "Aperture half-size; sets both x and y limits.",
    "X_LIMIT": "Horizontal aperture half-size.",
    "Y_LIMIT": "Vertical aperture half-size.",
    "X1_LIMIT": "Aperture limit on the -x side.",
    "X2_LIMIT": "Aperture limit on the +x side.",
    "Y1_LIMIT": "Aperture limit on the -y side.",
    "Y2_LIMIT": "Aperture limit on the +y side.",
    "APERTURE_AT": "Longitudinal location(s) where the aperture is applied.",
    "APERTURE_TYPE": "Aperture shape (rectangular, elliptical, ...).",
    "OFFSET_MOVES_APERTURE": "If True, element offsets shift the aperture too.",
    "WALL": "Vacuum chamber wall / aperture cross-section.",
    # Fringe fields
    "FRINGE_TYPE": "Type of fringe field to apply.",
    "FRINGE_AT": "Element end(s) at which fringe fields are applied.",
    # Description / label strings
    "TYPE": "User-defined type string.",
    "ALIAS": "User-defined alias name.",
    "DESCRIP": "User-defined description string.",
    # Superposition
    "SUPERIMPOSE": "If True, superimpose this element onto the lattice.",
    "OFFSET": "Longitudinal offset of the superposition reference point.",
    "REFERENCE": "Element used as the superposition reference.",
    "REF_ORIGIN": "Reference-element origin point for superposition.",
    "ELE_ORIGIN": "This element's origin point for superposition.",
    "CREATE_JUMBO_SLAVE": "If True, create a single jumbo super-slave.",
    "WRAP_SUPERIMPOSE": "If True, allow superposition to wrap around the ring.",
    # Field maps
    "CARTESIAN_MAP": "Field map defined in Cartesian coordinates.",
    "CYLINDRICAL_MAP": "Field map defined in cylindrical coordinates.",
    "GRID_FIELD": "Field defined on a grid of points.",
    "GEN_GRADIENTS": "Field defined by generalized gradients.",
    "FIELD_OVERLAPS": "Elements whose fields overlap this one.",
    # Wakefields
    "SR_WAKE": "Short-range wakefield definition.",
    "LR_WAKE": "Long-range wakefield definition.",
    "SR_WAKE_FILE": "File defining the short-range wakefield.",
    "LR_WAKE_FILE": "File defining the long-range wakefield.",
    "LR_SELF_WAKE_ON": "If True, include the long-range self-wake.",
    "LR_FREQ_SPREAD": "Fractional spread in long-range wake mode frequencies.",
}


def _add_multipole_families() -> None:
    """
    Populate the regular multipole coefficient families (a_n, b_n, k_nl, t_n).

    Enumerating ~90 near-identical entries by hand is noise; generate them.
    """
    for n in range(0, 22):
        COMMON.setdefault(f"A{n}", f"Skew magnetic multipole coefficient of order {n}.")
        COMMON.setdefault(f"B{n}", f"Normal magnetic multipole coefficient of order {n}.")
        COMMON.setdefault(f"A{n}_ELEC", f"Skew electric multipole coefficient of order {n}.")
        COMMON.setdefault(f"B{n}_ELEC", f"Normal electric multipole coefficient of order {n}.")
        COMMON.setdefault(f"K{n}L", f"Integrated normal multipole strength of order {n}.")
        COMMON.setdefault(f"T{n}", f"Tilt of the order-{n} multipole.")


_add_multipole_families()
