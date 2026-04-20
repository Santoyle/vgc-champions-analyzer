"""
CLI para regenerar el checksum SHA-256 de archivos JSON de regulación.

Tres modos de operación:

  Normal:
    python scripts/rehash.py regulations/M-A.json
    Recalcula checksum_sha256 y actualiza last_verified en cada archivo.

  Dry-run:
    python scripts/rehash.py --dry-run regulations/M-A.json
    Calcula el nuevo checksum y lo muestra en pantalla sin escribir nada.

  Verify:
    python scripts/rehash.py --verify regulations/M-A.json
    Solo valida que el checksum_sha256 almacenado coincide con el contenido
    actual, sin modificar ningún archivo.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.core.checksum import rehash_dict, verify_checksum


def rehash_file(path: Path, dry_run: bool = False) -> bool:
    """
    Procesa un archivo JSON de regulación:
    1. Lo carga desde disco.
    2. Actualiza last_verified a hoy.
    3. Recalcula checksum_sha256 con rehash_dict().
    4. Verifica que el nuevo checksum es válido.
    5. Sobreescribe el archivo con indentación de 2 espacios
       y ensure_ascii=False.

    Args:
        path: Path al archivo JSON de regulación.
        dry_run: Si True, calcula y muestra el nuevo checksum pero
                 NO sobreescribe el archivo.

    Returns:
        True si el proceso fue exitoso, False si hubo error.

    Prints:
        - "✓ {path}: checksum actualizado → {nuevo_hash[:8]}..."
          si dry_run=False y fue exitoso.
        - "~ {path}: checksum calculado → {nuevo_hash[:8]}..."
          si dry_run=True.
        - "✗ {path}: {mensaje de error}"
          si hubo cualquier error.
    """
    try:
        data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        data["last_verified"] = date.today().isoformat()

        data_rehashed = rehash_dict(data)
        new_hash: str = str(data_rehashed["checksum_sha256"])

        if not verify_checksum(data_rehashed, new_hash):
            raise RuntimeError(
                "El checksum recién calculado no pasa su propia verificación. "
                "Esto es un bug interno en checksum.py."
            )

        if dry_run:
            print(f"~ {path}: checksum calculado → {new_hash[:8]}...")
        else:
            path.write_text(
                json.dumps(data_rehashed, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"✓ {path}: checksum actualizado → {new_hash[:8]}...")

    except Exception as exc:  # noqa: BLE001
        print(f"✗ {path}: {exc}")
        return False

    return True


def main() -> int:
    """
    Entry point del CLI.

    Returns:
        0 si todos los archivos fueron procesados exitosamente,
        1 si alguno falló.
    """
    parser = argparse.ArgumentParser(
        description="Regenera o verifica checksums SHA-256 de JSONs de regulación.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="Uno o más archivos JSON de regulación.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calcular checksum sin sobreescribir el archivo.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Solo verificar checksums existentes sin actualizar.",
    )

    args = parser.parse_args()
    files: list[Path] = args.files

    if args.verify:
        all_valid = True
        for path in files:
            try:
                data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
                stored = str(data.get("checksum_sha256", ""))
                if verify_checksum(data, stored):
                    print(f"✓ {path}: válido")
                else:
                    print(f"✗ {path}: inválido")
                    all_valid = False
            except Exception as exc:  # noqa: BLE001
                print(f"✗ {path}: {exc}")
                all_valid = False
        return 0 if all_valid else 1

    errors = 0
    for path in files:
        if not rehash_file(path, dry_run=args.dry_run):
            errors += 1

    print(f"\nProcesados: {len(files)}, Errores: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
