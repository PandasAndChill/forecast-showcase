"""ptproxy — country-level powertrain mix proxy for markets without a direct forecast.

Public-safe showcase: every number this package produces comes from the
synthetic generator in :mod:`ptproxy.synthetic`.
"""

from ptproxy._const import POWERTRAINS
from ptproxy.allocate import allocate_region, country_tilts
from ptproxy.labels import confusion_from_reference, relabel

__all__ = [
    "POWERTRAINS",
    "allocate_region",
    "country_tilts",
    "confusion_from_reference",
    "relabel",
]
