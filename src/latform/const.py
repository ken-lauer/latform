from __future__ import annotations

import pathlib

from .token import Delimiter, Role

UNSET_PATH = pathlib.Path("unnamed")

NON_EXPR_DELIMITERS = frozenset("[:](,){}=&;")
EXPR_DELIMITERS = frozenset("+/-*^")
DELIMITERS = NON_EXPR_DELIMITERS | EXPR_DELIMITERS

STATEMENT_NAME_COLON = Delimiter(":", role=Role.statement_definition)
STATEMENT_NAME_EQUALS = Delimiter("=", role=Role.statement_definition)

COLON = Delimiter(":")
COMMA = Delimiter(",")
OPTIONAL_COMMA = Delimiter(",", role=Role.optional)
SLASH = Delimiter("/")
DQUOTE = Delimiter('"')
EQUALS = Delimiter("=")
LBRACE = Delimiter("{")
LBRACK = Delimiter("[")
LPAREN = Delimiter("(")
RBRACE = Delimiter("}")
RBRACK = Delimiter("]")
RPAREN = Delimiter(")")
SPACE = Delimiter(" ")
STAR = Delimiter("*")  # easier to type than asterisk
MINUS = Delimiter("-")
SQUOTE = Delimiter("'")
OPEN_TO_CLOSE = {
    "{": "}",
    "[": "]",
    "(": ")",
}
CLOSE_TO_OPEN = {v: Delimiter(k) for k, v in OPEN_TO_CLOSE.items()}


# From Bmad physical_constants.f90
# Updated to 2022 CODATA

pi = 3.141592653589793238462643383279
twopi = 2 * pi
fourpi = 4 * pi
sqrt_2 = 1.414213562373095048801688724209698
sqrt_3 = 1.732050807568877293527446341505872

m_electron = 0.51099895069e6  # Mass [eV]
m_proton = 0.93827208943e9  # Mass [eV]
m_neutron = 0.93956542194e9  # Mass [eV]
m_muon = 105.6583755e6  # Mass [eV]
m_helion = 2.80839161112e9  # Mass He3 nucleus

e_mass = 1e-9 * m_electron  # [GeV] FOR MAD COMPATIBILITY USE ONLY. USE M_ELECTRON INSTEAD.
p_mass = 1e-9 * m_proton  # [GeV] FOR MAD COMPATIBILITY USE ONLY. USE M_PROTON INSTEAD.

m_pion_0 = 134.9768e6  # Mass [eV]
m_pion_charged = 139.57039e6  # Mass [eV]

m_deuteron = 1.87561294500e9  # Mass [eV]

atomic_mass_unit = 931.49410372e6  # unified atomic mass unit u (or dalton) in [eV]

c_light = 2.99792458e8  # speed of light
r_e = 2.8179403227e-15  # classical electron radius
r_p = r_e * m_electron / m_proton  # proton radius
e_charge = 1.602176634e-19  # electron charge [Coul]
h_planck = 4.135667696e-15  # Planck's constant [eV*sec]
h_bar_planck = h_planck / twopi  # h_planck/twopi [eV*sec]

mu_0_vac = 1.25663706127e-6  # Vacuum permeability 2018 CODATA.
eps_0_vac = 1 / (c_light * c_light * mu_0_vac)  # Permittivity of free space

# # Radiation constants
#
classical_radius_factor = r_e * m_electron  # e^2 / (4 pi eps_0) [m*eV]
#  = classical_radius * mass * c^2.
# Is same for all particles of charge +/- 1.


# # Chemistry
#
# real(rp), parameter :: N_avogadro = 6.02214076e23    # Number / mole  (exact)
#
# Anomalous magnetic moment.
# Note: Deuteron g-factor
#   g_deu = (g_p / (mu_p / mu_N)) (mu_deu / mu_N) * (m_deu / m_p) * (q_p / q_deu) * (S_p / S_deu)
# The anomlous mag moment a = (g - 2) / 2 as always.

# For Helion:
#   g_eff = 2 * R_mass * (mu_h/mu_p) / Q
#         = 2 * (2808.39160743(85) MeV / 938.27208816(29) MeV) * (-2.127625307(25)) / (2)
#         = -6.368307373
#   anom_mag_moment = (g_eff - 2) / 2 = -4.184153686

fine_structure_constant = 7.2973525643e-3
anomalous_mag_moment_electron = 1.15965218059e-3
anomalous_mag_moment_proton = 1.79284734463
anomalous_mag_moment_muon = 1.1659217e-3  # ~fine_structure_constant / twopi
anomalous_mag_moment_deuteron = -0.14298726925
anomalous_mag_moment_neutron = -1.91304273
anomalous_mag_moment_He3 = -4.184153686

# Should make physical_const_list "parameter" but there is a gcc bug (in Version 7.1 at least)
# where if you pass physical_const_list%name to a routine there will be a crash.

named_physical_constants = {
    "pi": pi,
    "twopi": twopi,
    "fourpi": fourpi,
    "e_log": 2.71828182845904523,
    "e": 2.71828182845904523,
    "sqrt_2": sqrt_2,
    "degrad": 180 / pi,
    "degrees": pi / 180,  # From degrees to radians.
    "raddeg": pi / 180,
    "m_electron": m_electron,
    "m_muon": m_muon,
    "m_pion_0": m_pion_0,
    "m_pion_charged": m_pion_charged,
    "m_proton": m_proton,
    "m_deuteron": m_deuteron,
    "m_neutron": m_neutron,
    "c_light": c_light,
    "r_e": r_e,
    "r_p": r_p,
    "e_charge": e_charge,
    "h_planck": h_planck,
    "h_bar_planck": h_bar_planck,
    "pmass": p_mass,
    "emass": e_mass,
    "clight": c_light,
    "fine_struct_const": fine_structure_constant,
    "anom_moment_electron": anomalous_mag_moment_electron,
    "anom_moment_proton": anomalous_mag_moment_proton,
    "anom_moment_neutron": anomalous_mag_moment_neutron,
    "anom_moment_muon": anomalous_mag_moment_muon,
    "anom_moment_deuteron": anomalous_mag_moment_deuteron,
    "anom_moment_he3": anomalous_mag_moment_He3,
}
old_style_named_physical_constants = {
    "anom_mag_electron": anomalous_mag_moment_electron,  # Old style. Deprecated.
    "anom_mag_proton": anomalous_mag_moment_proton,  # Old style. Deprecated.
    "anom_mag_muon": anomalous_mag_moment_muon,  # Old style. Deprecated.
    "anom_mag_deuteron": anomalous_mag_moment_deuteron,  # Old style. Deprecated.
}
