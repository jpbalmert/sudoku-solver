class Cell:
    """Represents a single cell in the Sudoku grid."""

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        self.given: int | None = None
        self.solved: int | None = None
        self.candidates: set[int] = set()

    @property
    def value(self) -> int | None:
        """The cell's definite value (given or solved), or None if unsolved."""
        if self.given is not None:
            return self.given
        return self.solved

    @property
    def is_empty(self) -> bool:
        """True if the cell has no given or solved value."""
        return self.value is None

    @property
    def box_index(self) -> int:
        """The index (0-8) of the box this cell belongs to."""
        return (self.row // 3) * 3 + (self.col // 3)

    def __repr__(self) -> str:
        if self.given is not None:
            return f"Cell({self.row},{self.col} given={self.given})"
        if self.solved is not None:
            return f"Cell({self.row},{self.col} solved={self.solved})"
        if self.candidates:
            return f"Cell({self.row},{self.col} candidates={self.candidates})"
        return f"Cell({self.row},{self.col} empty)"
