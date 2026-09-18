from pathlib import Path

p = Path("src/mlcast/data/source_data_datasets.py")
s = p.read_text()


def replace_once(old: str, new: str, label: str) -> None:
    global s
    if old not in s:
        raise SystemExit(f"Could not find expected block for {label}:\n{old}")
    s = s.replace(old, new, 1)


# 1. Add normalization_names to the base dataset signature.
replace_once(
    """        storage_options: dict[str, Any] | None = None,
    ) -> None:
""",
    """        storage_options: dict[str, Any] | None = None,
        normalization_names: list[str] | None = None,
    ) -> None:
""",
    "base signature",
)

# 2. Add normalization_names to the precomputed dataset signature.
replace_once(
    """        time_depth: int = 24,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
""",
    """        time_depth: int = 24,
        storage_options: dict[str, Any] | None = None,
        normalization_names: list[str] | None = None,
    ) -> None:
""",
    "precomputed signature",
)

# 3. Add normalization_names to the random dataset signature.
replace_once(
    """        epoch_size: int = 1000,
        storage_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
""",
    """        epoch_size: int = 1000,
        storage_options: dict[str, Any] | None = None,
        normalization_names: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
""",
    "random signature",
)

# 4. Store and validate normalization_names in the base class.
replace_once(
    """        self.standard_names = standard_names
        self.input_steps = input_steps
""",
    """        self.standard_names = standard_names
        self.normalization_names = normalization_names or standard_names
        if len(self.normalization_names) != len(self.standard_names):
            raise ValueError(
                "normalization_names must have the same length as standard_names. "
                f"Got {len(self.normalization_names)} normalization names for "
                f"{len(self.standard_names)} standard names."
            )

        missing_normalization_names = [
            name for name in self.normalization_names if name not in NORMALIZATION_REGISTRY
        ]
        if missing_normalization_names:
            raise KeyError(
                "No normalization function registered for: "
                f"{missing_normalization_names}. Available names are: "
                f"{sorted(NORMALIZATION_REGISTRY)}"
            )

        self.input_steps = input_steps
""",
    "base normalization_names storage",
)

# 5. Pass normalization_names to the base class from both concrete datasets.
old_super = """            width=width,
            height=height,
            storage_options=storage_options,
        )
"""
new_super = """            width=width,
            height=height,
            storage_options=storage_options,
            normalization_names=normalization_names,
        )
"""

count = s.count(old_super)
if count != 2:
    raise SystemExit(f"Expected 2 super().__init__ blocks to update, found {count}.")
s = s.replace(old_super, new_super)

# 6. Update only the two normalization loops.
lines = s.splitlines()
norm_lines = [
    i for i, line in enumerate(lines)
    if "norm_func = NORMALIZATION_REGISTRY[std_name]" in line
]

if len(norm_lines) != 2:
    raise SystemExit(f"Expected 2 normalization lines, found {len(norm_lines)}: {norm_lines}")

for i in norm_lines:
    lines[i] = lines[i].replace(
        "NORMALIZATION_REGISTRY[std_name]",
        "NORMALIZATION_REGISTRY[norm_name]",
    )

    for j in range(i - 1, max(-1, i - 10), -1):
        if lines[j].strip() == "for std_name in self.standard_names:":
            indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
            lines[j] = (
                indent
                + "for std_name, norm_name in zip("
                + "self.standard_names, self.normalization_names, strict=True"
                + "):"
            )
            break
    else:
        raise SystemExit(f"Could not find loop header before line {i + 1}.")

p.write_text("\n".join(lines) + "\n")
print("Updated", p)
