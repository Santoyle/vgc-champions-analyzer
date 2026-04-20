"""
Funciones de hashing canónico para RegulationConfig.

Este módulo provee las utilidades necesarias para calcular, verificar y
regenerar el checksum SHA-256 de un diccionario de regulación. Trabaja
con dict[str, Any] en lugar de importar RegulationConfig directamente,
evitando importaciones circulares.

EXCLUDED_FROM_HASH define qué campos se omiten del hash y por qué:
- checksum_sha256: es el resultado del hash, no puede ser parte del input.
- last_verified: cambia en cada verificación sin que el contenido real de
  la regulación varíe; incluirla generaría hashes distintos sin razón
  semántica.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

EXCLUDED_FROM_HASH: frozenset[str] = frozenset(
    {
        "checksum_sha256",
        "last_verified",
    }
)


def compute_checksum(data: dict[str, Any]) -> str:
    """
    Calcula el SHA-256 canónico de un diccionario de regulación.

    Excluye siempre EXCLUDED_FROM_HASH antes de hashear. La serialización
    usa sort_keys=True y separators=(",", ":") para garantizar un output
    determinista independiente del orden de las claves.

    Args:
        data: Diccionario crudo de la regulación (puede incluir
              checksum_sha256 y last_verified; se excluyen internamente).

    Returns:
        Hexdigest SHA-256 de 64 caracteres en minúsculas.
    """
    payload: dict[str, Any] = deepcopy(data)
    for key in EXCLUDED_FROM_HASH:
        payload.pop(key, None)

    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(blob).hexdigest()


def verify_checksum(data: dict[str, Any], expected: str) -> bool:
    """
    Verifica que el checksum calculado coincide con el valor esperado.

    Args:
        data: Diccionario crudo de la regulación.
        expected: Valor de checksum_sha256 almacenado en el JSON.

    Returns:
        True si el checksum calculado coincide exactamente con expected,
        False en cualquier otro caso.
    """
    return compute_checksum(data) == expected


def rehash_dict(data: dict[str, Any]) -> dict[str, Any]:
    """
    Recalcula y actualiza el campo checksum_sha256 en un diccionario de
    regulación.

    NO modifica el diccionario original — retorna una copia con el
    checksum actualizado. Útil para scripts/rehash.py cuando se edita
    un JSON de regulación manualmente.

    Args:
        data: Diccionario crudo de la regulación con o sin
              checksum_sha256 previo.

    Returns:
        Nueva copia del diccionario con checksum_sha256 actualizado.
    """
    result: dict[str, Any] = deepcopy(data)
    result["checksum_sha256"] = compute_checksum(result)
    return result


__all__ = [
    "EXCLUDED_FROM_HASH",
    "compute_checksum",
    "verify_checksum",
    "rehash_dict",
]
