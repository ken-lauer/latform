import pytest

from ..species import (
    KNOWN_MOLECULES,
    SUBATOMIC_PARTICLES,
    Atom,
    GenericParticle,
    Molecule,
    ParticleSpeciesError,
    SubatomicParticle,
    parse_charge,
    parse_species,
    species_to_string,
)
from ..species import (
    _parse_mass as parse_mass,
)


@pytest.mark.parametrize(
    "charge_str,expected",
    [
        ("+", 1),
        ("-", -1),
        ("++", 2),
        ("--", -2),
        ("---", -3),
        ("+3", 3),
        ("-5", -5),
        ("", 0),
    ],
)
def test_parse_charge_valid(charge_str, expected):
    assert parse_charge(charge_str) == expected


@pytest.mark.parametrize(
    "charge_str,match",
    [
        ("+128", "out of valid range"),
        ("-128", "out of valid range"),
        ("abc", "Invalid charge"),
    ],
)
def test_parse_charge_invalid(charge_str, match):
    with pytest.raises(ParticleSpeciesError, match=match):
        parse_charge(charge_str)


@pytest.mark.parametrize(
    "mass_str,expected",
    [
        ("12", 12.0),
        ("37.54", 37.54),
        ("0.5", 0.5),
    ],
)
def test_valid_mass(mass_str, expected):
    assert parse_mass(mass_str) == expected


def test_negative_mass():
    with pytest.raises(ParticleSpeciesError, match="cannot be negative"):
        parse_mass("-1.0")


def test_invalid_mass():
    with pytest.raises(ParticleSpeciesError, match="Invalid mass"):
        parse_mass("abc")


@pytest.mark.parametrize(
    "input_name,expected_type,expected_charge,expected_name",
    [
        ("electron", SubatomicParticle, -1, "Electron"),
        ("positron", SubatomicParticle, 1, None),
        ("proton", SubatomicParticle, 1, None),
        ("antiproton", SubatomicParticle, -1, None),
        ("neutron", SubatomicParticle, 0, None),
        ("photon", SubatomicParticle, 0, None),
        ("muon", SubatomicParticle, -1, None),
        ("antimuon", SubatomicParticle, 1, None),
        ("pion+", SubatomicParticle, 1, None),
        ("pion-", SubatomicParticle, -1, None),
        ("pion0", SubatomicParticle, 0, None),
        ("deuteron", SubatomicParticle, None, None),
        ("helion", SubatomicParticle, 2, None),
        ("ref_particle", SubatomicParticle, None, None),
    ],
)
def test_subatomic_particles(input_name, expected_type, expected_charge, expected_name):
    result = parse_species(input_name)
    assert isinstance(result, expected_type)
    if expected_charge is not None:
        assert result.charge == expected_charge
    if expected_name is not None:
        assert result.name == expected_name


@pytest.mark.parametrize("input_name", ["ELECTRON", "Electron", "electron"])
def test_case_insensitive(input_name):
    result = parse_species(input_name)
    assert result.name == "Electron"


@pytest.mark.parametrize(
    "input_name,expected_name",
    [
        ("PION_PLUS", "Pion+"),
        ("pion_minus", "Pion-"),
        ("PION_0", "Pion0"),
    ],
)
def test_old_style_pion(input_name, expected_name):
    result = parse_species(input_name)
    assert isinstance(result, SubatomicParticle)
    assert result.name == expected_name


def test_ref_species_alias():
    result = parse_species("REF_SPECIES")
    assert isinstance(result, SubatomicParticle)
    assert result.name == "Ref_Particle"


@pytest.mark.parametrize(
    "species_str,element,charge,nucleon_number,is_anti",
    [
        ("C", "C", 0, None, False),
        ("C+", "C", 1, None, False),
        ("C+3", "C", 3, None, False),
        ("He--", "He", -2, None, False),
        ("#12C+3", "C", 3, 12, False),
        ("#C", "C", None, None, False),
        ("Au-79", "Au", -79, None, False),
        ("antiAu-79", "Au", -79, None, True),
        ("H+", "H", 1, None, False),
        ("U", "U", 0, None, False),
    ],
)
def test_atom_parsing(species_str, element, charge, nucleon_number, is_anti):
    result = parse_species(species_str)
    assert isinstance(result, Atom)
    assert result.element == element
    if charge is not None:
        assert result.charge == charge
    if nucleon_number is not None:
        assert result.nucleon_number == nucleon_number
    if is_anti:
        assert result.is_anti is True


def test_atom_case_sensitive():
    """Atoms are case sensitive"""
    with pytest.raises(ParticleSpeciesError):
        parse_species("c")  # lowercase 'c' is not valid


def test_atom_mass_not_allowed():
    with pytest.raises(ParticleSpeciesError, match="not allowed for atoms"):
        parse_species("C@M12.0")


@pytest.mark.parametrize(
    "species_str,expected_formula,expected_charge,expected_mass",
    [
        ("H2O", "H2O", 0, None),
        ("H2O+", "H2O", 1, None),
        ("NH3", "NH3", 0, None),
        ("CH2", "CH2", 0, None),
        ("CH3++", "CH3", 2, None),
        ("CH3+2", "CH3", 2, None),
        ("C2H3@M28.4+", "C2H3", 1, 28.4),
        ("CO2", "CO2", 0, None),
        ("D2O", "D2O", 0, None),
        ("HF", "HF", 0, None),
    ],
)
def test_molecule_parsing(species_str, expected_formula, expected_charge, expected_mass):
    result = parse_species(species_str)
    assert isinstance(result, Molecule)
    assert result.formula == expected_formula
    assert result.charge == expected_charge
    assert result.mass_amu == expected_mass


@pytest.mark.parametrize("mol", KNOWN_MOLECULES)
def test_all_known_molecules(mol):
    result = parse_species(mol)
    assert isinstance(result, Molecule)
    assert result.formula == mol


def test_molecule_case_sensitive():
    """Molecules are case sensitive"""
    with pytest.raises(ParticleSpeciesError):
        parse_species("h2o")  # lowercase not valid


def test_molecule_isotope_not_allowed():
    with pytest.raises(ParticleSpeciesError, match="not allowed for molecules"):
        parse_species("#12H2O")


@pytest.mark.parametrize(
    "input_string,expected_mass,expected_charge",
    [
        ("@M37.54", 37.54, 0),
        ("@M37.54++", 37.54, 2),
        ("@M100.5-", 100.5, -1),
        ("@M50.0+5", 50.0, 5),
        ("@M20", 20.0, 0),
    ],
)
def test_parse_species_mass(input_string, expected_mass, expected_charge):
    """Tests for generic particles with specified mass and charge."""
    result = parse_species(input_string)
    assert isinstance(result, GenericParticle)
    assert result.mass_amu == expected_mass
    assert result.charge == expected_charge


@pytest.mark.parametrize(
    "species, expected",
    [
        (SUBATOMIC_PARTICLES["Electron"], "Electron"),
        (Atom("C", 1, None, False), "C+"),
        (Atom("C", 3, 12, False), "#12C+3"),
        (Atom("He", 0, None, False), "He"),
        (Atom("He", -2, None, False), "He--"),
        (Atom("Au", -79, None, True), "antiAu-79"),
        (Molecule("H2O", 1, None, True), "H2O+"),
        (Molecule("C2H3", 1, 28.4, True), "C2H3@M28.4+"),
        (Molecule("CO2", 0, None, True), "CO2"),
        (GenericParticle(37.54, 2), "@M37.54++"),
        (GenericParticle(50.0, 0), "@M50"),
    ],
)
def test_species_to_string(species, expected):
    assert species_to_string(species) == expected


@pytest.mark.parametrize(
    "species_str",
    [
        "electron",
        "proton",
        "C+",
        "#12C+3",
        "He--",
        "H2O",
        "H2O+",
        "NH3++",
        "@M37.54",
        "@M37.54++",
        "antiAu-79",
    ],
)
def test_round_trip(species_str):
    """Test that parsing and converting back gives the same string."""
    parsed = parse_species(species_str)
    reconstructed = species_to_string(parsed)
    reparsed = parse_species(reconstructed)
    assert parsed == reparsed


def test_edge_case_empty_string():
    with pytest.raises(ParticleSpeciesError, match="Empty species name"):
        parse_species("")


def test_edge_case_whitespace_only():
    with pytest.raises(ParticleSpeciesError, match="Empty species name"):
        parse_species("   ")


def test_edge_case_invalid_element():
    with pytest.raises(ParticleSpeciesError, match="Cannot parse"):
        parse_species("Xx")


def test_edge_case_invalid_molecule():
    with pytest.raises(ParticleSpeciesError, match="Cannot parse"):
        parse_species("XYZ")


def test_edge_case_mass_without_m():
    """@M format requires the M"""
    with pytest.raises(ParticleSpeciesError):
        parse_species("@37.54")


def test_edge_case_mixed_case_molecule():
    """Molecules are case sensitive"""
    with pytest.raises(ParticleSpeciesError):
        parse_species("h2O")  # Wrong case


def test_edge_case_charge_at_beginning():
    """Charge must come at the end"""
    with pytest.raises(ParticleSpeciesError):
        parse_species("+C")


def test_edge_case_multiple_at_signs():
    """Only one @M allowed"""
    result = parse_species("@M37.5")
    assert isinstance(result, GenericParticle)
    # Having multiple @M would be caught in regex


@pytest.mark.parametrize(
    "input_str,expected_type,expected_attrs",
    [
        ("#12C+3", Atom, {"element": "C", "nucleon_number": 12, "charge": 3}),
        ("He--", Atom, {"element": "He", "charge": -2}),
        ("C2H3@M28.4+", Molecule, {"formula": "C2H3", "mass_amu": 28.4, "charge": 1}),
        ("CH2", Molecule, {"formula": "CH2", "charge": 0}),
        ("@M37.54++", GenericParticle, {"mass_amu": 37.54, "charge": 2}),
    ],
)
def test_doc_examples(input_str, expected_type, expected_attrs):
    result = parse_species(input_str)
    assert isinstance(result, expected_type)
    for attr, value in expected_attrs.items():
        assert getattr(result, attr) == value


@pytest.mark.parametrize("element", ["Uut", "Uup", "Uus", "Uuo"])
def test_special_elements(element):
    result = parse_species(element)
    assert isinstance(result, Atom)
    assert result.element == element
