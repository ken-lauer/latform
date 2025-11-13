import re
from dataclasses import dataclass

from .const import (
    m_deuteron,
    m_electron,
    m_helion,
    m_muon,
    m_neutron,
    m_pion_0,
    m_pion_charged,
    m_proton,
)
from .exceptions import InvalidChargeError, ParticleSpeciesError

# TODO add to SubatomicParticle
# anomalous_moment_of_subatomi = [
#     "anomalous_mag_moment_He3",
#     0.0,
#     "anomalous_mag_moment_neutron",
#     "anomalous_mag_moment_deuteron",
#     0.0,
#     "anomalous_mag_moment_muon",
#     "anomalous_mag_moment_proton",
#     "anomalous_mag_moment_electron",
#     0.0,
#     "anomalous_mag_moment_electron",
#     "anomalous_mag_moment_proton",
#     "anomalous_mag_moment_muon",
#     0.0,
#     "anomalous_mag_moment_deuteron",
#     "anomalous_mag_moment_neutron",
#     0.0,
#     "anomalous_mag_moment_He3",
#     0.0,
# ]

# !----------------------
# ! Atoms
#
ELEMENT_SYMBOLS = """\
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc
Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo
Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu
Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po
At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db
Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
""".strip().split()


# !----------------------
# ! Known Molecules
#
KNOWN_MOLECULES = [
    "CH2",
    "CH3",
    "CH4",
    "CO",
    "CO2",
    "D2",
    "D2O",
    "H2",
    "H2O",
    "N2",
    "HF",
    "OH",
    "O2",
    "NH2",
    "NH3",
    "C2H3",
    "C2H4",
    "C2H5",
]

molecular_mass = [
    14.026579004642441,  # for CH2
    15.034498933118178,  # for CH3
    16.04249886158824,  # for CH4
    28.01009801234052,  # for CO
    44.009496876987235,  # for CO2
    4.028203269749646,  # for D2
    20.02759857879661,  # for D2O
    2.015879856948637,  # for H2
    18.01527872159535,  # for H2O
    28.013398012106343,  # for N2
    20.006341580305058,  # for HF
    17.00733879312103,  # for OH
    31.998797729293425,  # for O2
    16.0228,
    17.030518791476126,  # for NH2, NH3
    27.0457,
    28.0536,
    29.0615,  # For C2H3, C2H4, C2H5
]


@dataclass(frozen=True)
class SubatomicParticle:
    """
    Subatomic particle species.

    Parameters
    ----------
    name : str
        Canonical name of the subatomic particle (e.g., 'electron', 'proton').
    pmd_name : str
        OpenPMD name.
    charge : int
        Particle charge in units of elementary charge.
    mass : float
        Particle mass in eV/c².
    spin : float
    antiparticle : str
    """

    name: str
    pmd_name: str
    charge: int
    mass: float
    spin: float
    antiparticle: str


@dataclass(frozen=True)
class Atom:
    """
    Atomic species.

    Parameters
    ----------
    element : str
        Chemical symbol (e.g., 'C', 'He', 'Au').
    charge : int
        Ionic charge in units of elementary charge.
    nucleon_number : int or None
        Number of nucleons (mass number). If None, uses average isotopic mass.
    is_anti : bool
        Whether this is an anti-atom, default False.
    """

    element: str
    charge: int
    nucleon_number: int | None = None
    is_anti: bool = False


@dataclass(frozen=True)
class Molecule:
    """
    Molecular species.

    Parameters
    ----------
    formula : str
        Molecular formula (e.g., 'H2O', 'NH3').
    charge : int
        Molecular charge in units of elementary charge.
    mass_amu : float or None
        Specified mass in atomic mass units. If None, uses default molecular mass.
    is_known : bool
        Whether this is a known molecule with a predefined formula.
    """

    formula: str
    charge: int
    mass_amu: float | None = None
    is_known: bool = True


@dataclass(frozen=True)
class GenericParticle:
    """
    Generic particle with only mass and charge specified.

    Parameters
    ----------
    mass_amu : float
        Particle mass in atomic mass units.
    charge : int
        Particle charge in units of elementary charge.
    """

    mass_amu: float
    charge: int


# Type alias for any particle species
ParticleSpecies = SubatomicParticle | Atom | Molecule | GenericParticle

# Subatomic particle definitions

SUBATOMIC_PARTICLES = {
    "Anti_Helion": SubatomicParticle("Anti_Helion", "Garbage!", -2, m_helion, 0.0, "Helion"),
    "Anti_Ref_Particle": SubatomicParticle(
        "Anti_Ref_Particle", "Garbage!", 0, 0.0, -987654.3, "Ref_Particle"
    ),
    "Anti_Neutron": SubatomicParticle("Anti_Neutron", "anti-neutron", 0, m_neutron, 0.5, "Neutron"),
    "Anti_Deuteron": SubatomicParticle(
        "Anti_Deuteron", "anti-deuteron", -1, m_deuteron, 1.0, "Deuteron"
    ),
    "Pion-": SubatomicParticle("Pion-", "pion-", -1, m_pion_charged, 0.0, "Pion+"),
    "Muon": SubatomicParticle("Muon", "muon", -1, m_muon, 0.5, "Antimuon"),
    "Antiproton": SubatomicParticle("Antiproton", "anti-proton", -1, m_proton, 0.5, "Proton"),
    "Electron": SubatomicParticle("Electron", "electron", -1, m_electron, 0.5, "Positron"),
    "Photon": SubatomicParticle("Photon", "photon", 0, 0.0, 0.0, "Photon"),
    "Positron": SubatomicParticle("Positron", "positron", 1, m_electron, 0.5, "Electron"),
    "Proton": SubatomicParticle("Proton", "proton", 1, m_proton, 0.5, "Antiproton"),
    "Antimuon": SubatomicParticle("Antimuon", "anti-muon", 1, m_muon, 0.5, "Muon"),
    "Pion+": SubatomicParticle("Pion+", "pion+", 1, m_pion_charged, 0.0, "Pion-"),
    "Deuteron": SubatomicParticle("Deuteron", "deuteron", 1, m_deuteron, 1.0, "Anti_Deuteron"),
    "Neutron": SubatomicParticle("Neutron", "neutron", 0, m_neutron, 0.5, "Anti_Neutron"),
    "Ref_Particle": SubatomicParticle(
        "Ref_Particle", "Garbage!", 0, 0.0, -987654.3, "Anti_Ref_Particle"
    ),
    "Helion": SubatomicParticle("Helion", "Garbage!", 2, m_helion, 0.0, "Anti_Helion"),
    "Pion0": SubatomicParticle("Pion0", "pion0", 0, m_pion_0, 0.0, "Pion0"),
}


LOWER_SUBATOMIC_PARTICLES = {key.lower(): value for key, value in SUBATOMIC_PARTICLES.items()}


def parse_charge(charge_str: str) -> int:
    """
    Parse charge specification from string.

    Parameters
    ----------
    charge_str : str
        Charge specification (e.g., '+', '++', '-3', '+2').

    Returns
    -------
    int
    """
    if not charge_str:
        return 0

    if all(c == "+" for c in charge_str):
        charge = len(charge_str)
    elif all(c == "-" for c in charge_str):
        charge = -len(charge_str)
    elif charge_str[0] in "+-":
        # +N or -N format
        try:
            charge = int(charge_str)
        except ValueError:
            raise InvalidChargeError(f"Invalid charge specification: {charge_str}")
    else:
        raise InvalidChargeError(f"Invalid charge specification: {charge_str}")

    if not -127 <= charge <= 127:
        raise ParticleSpeciesError(f"Charge {charge} out of valid range [-127, 127]")

    return charge


def _parse_mass(mass_str: str) -> float:
    """
    Parse mass specification from @M format.

    Parameters
    ----------
    mass_str : str
        Mass specification after '@M' (e.g., '12.01', '37.5').

    Returns
    -------
    float
        Mass in atomic mass units.
    """
    try:
        mass = float(mass_str)
        if mass < 0:
            raise ParticleSpeciesError(f"Mass cannot be negative: {mass}")
        return mass
    except ValueError:
        raise ParticleSpeciesError(f"Invalid mass specification: {mass_str}")


def parse_species(name: str) -> ParticleSpecies:
    """
    Parse a particle species name into its structured representation.

    Parameters
    ----------
    name : str
        Particle species name following Bmad conventions.

    Returns
    -------
    ParticleSpecies
        Parsed particle species (SubatomicParticle, Atom, Molecule, or GenericParticle).

    Raises
    ------
    ParticleSpeciesError
        If the species name cannot be parsed or is invalid.

    Examples
    --------
    >>> parse_species('electron')
    SubatomicParticle(name='electron', species_id=-1, charge=-1, mass=0.5109989461)

    >>> parse_species('#12C+3')
    Atom(element='C', charge=3, nucleon_number=12, is_anti=False)

    >>> parse_species('H2O+')
    Molecule(formula='H2O', charge=1, mass_amu=None, is_known=True)

    >>> parse_species('@M37.54++')
    GenericParticle(mass_amu=37.54, charge=2)
    """
    if not name or not name.strip():
        raise ParticleSpeciesError("Empty species name")

    original_name = name
    name = name.strip()

    # Handle special cases
    if name.upper() == "REF_SPECIES":
        return LOWER_SUBATOMIC_PARTICLES["ref_particle"]

    # Check for subatomic particle (case insensitive)
    subatomic_key = name.lower()

    subatomic_key = {
        "pion_plus": "pion+",
        "pion_0": "pion0",
        "pion_minus": "pion-",
    }.get(subatomic_key, subatomic_key)

    if subatomic_key in LOWER_SUBATOMIC_PARTICLES:
        return LOWER_SUBATOMIC_PARTICLES[subatomic_key]

    # Parse components: mass (@M), charge (+/-), isotope (#), anti prefix
    remaining = name

    # Extract mass specification (@M)
    mass_amu: float | None = None
    mass_match = re.search(r"@M([\d.]+)", remaining)
    if mass_match:
        mass_amu = _parse_mass(mass_match.group(1))
        remaining = remaining[: mass_match.start()] + remaining[mass_match.end() :]

    # Extract charge specification
    charge = 0
    # Find rightmost + or - that starts a charge specification
    charge_match = re.search(r"([+-](?:\d+|[+-]*))$", remaining)
    if charge_match:
        charge_str = charge_match.group(1)
        # Only parse if there's content after the first +/-
        if len(charge_str) > 0:
            charge = parse_charge(charge_str)
            remaining = remaining[: charge_match.start()]

    # Check for anti prefix
    is_anti = False
    if remaining.startswith("anti"):
        is_anti = True
        remaining = remaining[4:]

    # Extract isotope number (#NNN)
    nucleon_number: int | None = None
    if remaining.startswith("#"):
        iso_match = re.match(r"#(\d*)", remaining)
        if iso_match:
            iso_str = iso_match.group(1)
            nucleon_number = int(iso_str) if iso_str else None
            remaining = remaining[iso_match.end() :]

    # Now determine what type of particle we have
    if not remaining:
        # Only mass and charge specified -> GenericParticle
        if mass_amu is not None:
            return GenericParticle(mass_amu=mass_amu, charge=charge)
        else:
            raise ParticleSpeciesError(f"Cannot determine particle type from: {original_name}")

    # Check if it's a known molecule
    if remaining in KNOWN_MOLECULES:
        if nucleon_number is not None:
            raise ParticleSpeciesError(f"Isotope number not allowed for molecules: {original_name}")
        if is_anti:
            raise ParticleSpeciesError(f"Anti prefix not allowed for molecules: {original_name}")
        return Molecule(formula=remaining, charge=charge, mass_amu=mass_amu, is_known=True)

    # Check if it's an atom
    if remaining in ELEMENT_SYMBOLS or remaining in {"Uut", "Uup", "Uus", "Uuo"}:
        if mass_amu is not None:
            raise ParticleSpeciesError(f"Mass specification not allowed for atoms: {original_name}")
        return Atom(
            element=remaining,
            charge=charge,
            nucleon_number=nucleon_number,
            is_anti=is_anti,
        )

    # If we have a mass specification and remaining text, it's an error
    if mass_amu is not None and remaining:
        raise ParticleSpeciesError(f"Cannot parse species name: {original_name}")

    # Must be an unknown molecule with only mass
    if mass_amu is not None:
        return GenericParticle(mass_amu=mass_amu, charge=charge)

    raise ParticleSpeciesError(f"Cannot parse species name: {original_name}")


def species_to_string(species: ParticleSpecies) -> str:
    """
    Convert a particle species object back to its string representation.

    Parameters
    ----------
    species : ParticleSpecies
        Particle species object.

    Returns
    -------
    str

    Examples
    --------
    >>> species_to_string(SubatomicParticle('electron', -1, -1, 0.511))
    'electron'

    >>> species_to_string(Atom('C', 3, 12, False))
    '#12C+3'

    >>> species_to_string(Molecule('H2O', 1, None, True))
    'H2O+'
    """
    match species:
        case SubatomicParticle(name=name):
            return name

        case Atom(element=element, charge=charge, nucleon_number=n_nuc, is_anti=anti):
            result = ""
            if anti:
                result += "anti"
            if n_nuc is not None:
                result += f"#{n_nuc}"
            result += element
            result += _format_charge(charge)
            return result

        case Molecule(formula=formula, charge=charge, mass_amu=mass, is_known=_):
            result = formula
            if mass is not None:
                result += f"@M{mass:g}"
            result += _format_charge(charge)
            return result

        case GenericParticle(mass_amu=mass, charge=charge):
            result = f"@M{mass:g}"
            result += _format_charge(charge)
            return result


def _format_charge(charge: int) -> str:
    """
    Format charge as string suffix.

    Parameters
    ----------
    charge : int

    Returns
    -------
    str
        Formatted charge string (e.g., '+', '++', '-3', '+2', '').
    """
    if charge == 0:
        return ""
    if charge == 1:
        return "+"
    if charge == -1:
        return "-"
    if charge == 2:
        return "++"
    if charge == -2:
        return "--"
    if charge > 0:
        return f"+{charge}"
    return str(charge)
