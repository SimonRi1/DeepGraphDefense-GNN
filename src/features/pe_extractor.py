from __future__ import annotations
import lief
import numpy as np
from pathlib import Path


class PEFeatureExtractor:
    """
    Extraction of the 9 feature from a PE binary using LIEF.
    Output: Separated features for the building of the feature graph
    Note: No concatenation of the feature (EMBER approach)
    """

    def extract(self, pe_path: str) -> dict | None:
        try:
            binary = lief.parse(pe_path)
            if binary is None:
                return None
        except Exception:
            return None

        return {
            "general":         self._extract_general(binary),
            "header":          self._extract_header(binary),
            "imported":        self._extract_imported(binary),
            "exported":        self._extract_exported(binary),
            "section":         self._extract_section(binary),
            "byte_histogram":  self._extract_byte_histogram(pe_path),
            "byte_entropy":    self._extract_byte_entropy(pe_path),
            "data_directories":self._extract_data_directories(binary),
            "string":          self._extract_string(pe_path),
        }

    def _extract_general(self, binary) -> np.ndarray:
        # header = binary.header
        # opt = binary.optional_header
        return np.array([
            binary.virtual_size,
            binary.sizeof_headers,
            int(binary.has_debug),
            int(binary.has_tls),
            int(binary.has_resources),
            int(binary.has_relocations),
            int(binary.has_signature),
            len(list(binary.symbols)),
            len(list(binary.imported_functions)),
            len(list(binary.exported_functions)),
        ], dtype=np.float32)

    def _extract_header(self, binary) -> np.ndarray:
        header = binary.header
        opt = binary.optional_header
        return np.array([
            header.time_date_stamps,
            opt.major_image_version,
            opt.minor_image_version,
            opt.major_linker_version,
            opt.minor_linker_version,
            opt.major_operating_system_version,
            opt.minor_operating_system_version,
            opt.sizeof_code,
            opt.sizeof_initialized_data,
            opt.sizeof_uninitialized_data,
        ], dtype=np.float32)

    def _extract_imported(self, binary) -> np.ndarray:
        """Number of imported function per DLL."""
        imports = list(binary.imports)
        dll_counts = [len(list(imp.entries)) for imp in imports]
        if not dll_counts:
            return np.zeros(10, dtype=np.float32)
        arr = np.array(dll_counts[:10], dtype=np.float32)
        # Pad a lunghezza fissa
        return np.pad(arr, (0, max(0, 10 - len(arr))))

    def _extract_exported(self, binary) -> np.ndarray:
        exports = list(binary.exported_functions)
        return np.array([len(exports)], dtype=np.float32)

    def _extract_section(self, binary) -> np.ndarray:
        sections = list(binary.sections)
        if not sections:
            return np.zeros(10, dtype=np.float32)
        features = []
        for sec in sections[:5]:  # max 5 sezioni
            features.extend([
                sec.virtual_size,
                sec.size,
                sec.entropy,
            ])
        arr = np.array(features, dtype=np.float32)
        return np.pad(arr, (0, max(0, 15 - len(arr))))

    def _extract_byte_histogram(self, pe_path: str) -> np.ndarray:
        """Distribuzione dei 256 valori di byte nel file."""
        with open(pe_path, 'rb') as f:
            data = f.read()
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        return counts.astype(np.float32) / max(len(data), 1)

    def _extract_byte_entropy(self, pe_path: str,
                               window=1024, step=256) -> np.ndarray:
        """Entropia calcolata su finestre scorrevoli di byte."""
        with open(pe_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        entropies = []
        for i in range(0, len(data) - window, step):
            window_data = data[i:i+window]
            counts = np.bincount(window_data, minlength=256)
            probs = counts / window
            # Entropia di Shannon
            entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
            entropies.append(entropy)
        if not entropies:
            return np.zeros(256, dtype=np.float32)
        arr = np.array(entropies[:256], dtype=np.float32)
        return np.pad(arr, (0, max(0, 256 - len(arr))))

    def _extract_data_directories(self, binary) -> np.ndarray:
        dirs = list(binary.data_directories)
        features = []
        for d in dirs[:16]:  # 16 data directories standard PE
            features.extend([d.rva, d.size])
        arr = np.array(features, dtype=np.float32)
        return np.pad(arr, (0, max(0, 32 - len(arr))))

    def _extract_string(self, pe_path: str,
                         min_len: int = 4) -> np.ndarray:
        """Statistiche sulle stringhe ASCII nel file."""
        with open(pe_path, 'rb') as f:
            data = f.read()
        strings = []
        current = []
        for byte in data:
            if 0x20 <= byte <= 0x7e:
                current.append(chr(byte))
            else:
                if len(current) >= min_len:
                    strings.append(''.join(current))
                current = []
        if not strings:
            return np.zeros(6, dtype=np.float32)
        lengths = [len(s) for s in strings]
        return np.array([
            len(strings),                    # numero totale stringhe
            np.mean(lengths),                # lunghezza media
            np.std(lengths),                 # deviazione standard
            max(lengths),                    # stringa più lunga
            sum(1 for s in strings if '\\' in s),   # path-like
            sum(1 for s in strings if '://' in s),  # URL-like
        ], dtype=np.float32)