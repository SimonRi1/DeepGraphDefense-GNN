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
        return np.array([
            binary.virtual_size,
            binary.sizeof_headers,
            int(binary.has_debug),
            int(binary.has_tls),
            int(binary.has_resources),
            int(binary.has_relocations),
            int(binary.has_signatures),
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
        """Distribution of the 256 byte values in the file."""
        with open(pe_path, 'rb') as f:
            data = f.read()
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        return counts.astype(np.float32) / max(len(data), 1)

    def _extract_byte_entropy(self, pe_path: str, window=1024, step=256) -> np.ndarray:
        """Entropy calculated on sliding windows of bytes."""
        with open(pe_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        entropies = []
        for i in range(0, len(data) - window, step):
            window_data = data[i:i+window]
            counts = np.bincount(window_data, minlength=256)
            probs = counts / window
            # Shannon's entropy
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

    def _extract_string(self, pe_path: str, min_len: int = 4) -> np.ndarray:
        """Statistics on ASCII strings in the file."""
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
            len(strings),                    # Strings total number
            np.mean(lengths),                # avarage length
            np.std(lengths),                 # standard deviation
            max(lengths),                    # longer string
            sum(1 for s in strings if '\\' in s),   # path-like
            sum(1 for s in strings if '://' in s),  # URL-like
        ], dtype=np.float32)
    
class EmberFeatureParser:
    """
    Parses EMBER's raw jsonl schema into the exact same per-group array format
    as PEFeatureExtractor, allowing seamless integration with FeatureGraphBuilder.
    """

    def parse(self, sample: dict) -> dict:
        return {
            "general": self._parse_general(sample),
            "header": self._parse_header(sample["header"]),
            "imported": self._parse_imported(sample["imports"]),
            "exported": self._parse_exported(sample["exports"]),
            "section": self._parse_section(sample["section"]),
            "byte_histogram": np.array(sample["histogram"], dtype=np.float32),
            "byte_entropy": np.array(sample["byteentropy"], dtype=np.float32),
            "data_directories": self._parse_datadirs(sample["datadirectories"]),
            "string": self._parse_strings(sample["strings"]),
        }

    def _parse_general(self, sample: dict) -> np.ndarray:
        g = sample["general"]
        # sizeof_headers is stored inside header["optional"] in EMBER JSON
        sizeof_headers = sample["header"]["optional"].get("sizeof_headers", 0) 
        
        return np.array([
            g.get("vsize", 0), 
            sizeof_headers, 
            g.get("has_debug", 0), 
            g.get("has_tls", 0),
            g.get("has_resources", 0), 
            g.get("has_relocations", 0), 
            g.get("has_signatures", 0),
            g.get("symbols", 0), 
            g.get("imports", 0), 
            g.get("exports", 0),
        ], dtype=np.float32)

    def _parse_header(self, h: dict) -> np.ndarray:
        coff = h.get("coff", {})
        opt = h.get("optional", {})
        return np.array([
            coff.get("timestamp", 0),
            opt.get("major_image_version", 0),
            opt.get("minor_image_version", 0),
            opt.get("major_linker_version", 0),
            opt.get("minor_linker_version", 0),
            opt.get("major_operating_system_version", 0),
            opt.get("minor_operating_system_version", 0),
            opt.get("sizeof_code", 0),
            opt.get("sizeof_initialized_data", 0),
            opt.get("sizeof_uninitialized_data", 0),
        ], dtype=np.float32)

    def _parse_imported(self, imp: dict) -> np.ndarray:
        # EMBER saves imports as {"dll_name": ["func1", "func2"]}
        dll_counts = [len(funcs) for dll, funcs in imp.items()]
        if not dll_counts:
            return np.zeros(10, dtype=np.float32)
        arr = np.array(dll_counts[:10], dtype=np.float32)
        return np.pad(arr, (0, max(0, 10 - len(arr))))

    def _parse_exported(self, exp: list) -> np.ndarray:
        # EMBER saves exports as a list of strings
        return np.array([len(exp)], dtype=np.float32)

    def _parse_section(self, sec: dict) -> np.ndarray:
        # EMBER section features are within a nested list
        sections = sec.get("sections", [])
        if not sections:
            return np.zeros(15, dtype=np.float32)
        features = []
        for s in sections[:5]:
            features.extend([
                s.get("vsize", 0),
                s.get("size", 0),
                s.get("entropy", 0.0),
            ])
        arr = np.array(features, dtype=np.float32)
        return np.pad(arr, (0, max(0, 15 - len(arr))))

    def _parse_datadirs(self, dirs: list) -> np.ndarray:
        features = []
        for d in dirs[:16]:
            features.extend([d.get("virtual_address", 0), d.get("size", 0)])
        arr = np.array(features, dtype=np.float32)
        return np.pad(arr, (0, max(0, 32 - len(arr))))

    def _parse_strings(self, s: dict) -> np.ndarray:
        # Note: EMBER JSON does not contain max length or standard deviation. 
        # We fill with 0 to keep the vector shape (6,) compatible with PEFeatureExtractor.
        return np.array([
            s.get("numstrings", 0),
            s.get("avlength", 0.0),
            0.0,  # Missing std deviation
            0.0,  # Missing max length
            s.get("paths", 0),
            s.get("urls", 0),
        ], dtype=np.float32)