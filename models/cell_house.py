from abc import ABC, abstractmethod
from .cell import Cell


class CellHouse(ABC):
    """Interface for a group of 9 cells that must contain digits 1-9 exactly once."""

    def __init__(self, index: int, cells: list[Cell]):
        self.index = index
        self.cells = cells

    @property
    def solved_values(self) -> set[int]:
        """The set of digits already placed (given or solved) in this house."""
        return {c.value for c in self.cells if c.value is not None}

    @property
    def unsolved_cells(self) -> list[Cell]:
        """Cells in this house that don't yet have a value."""
        return [c for c in self.cells if c.is_empty]

    @property
    @abstractmethod
    def house_type(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def __repr__(self) -> str:
        return self.name


class CellRow(CellHouse):
    @property
    def house_type(self) -> str:
        return "row"

    @property
    def name(self) -> str:
        return f"Row {self.index + 1}"


class CellColumn(CellHouse):
    @property
    def house_type(self) -> str:
        return "column"

    @property
    def name(self) -> str:
        return f"Column {self.index + 1}"


class CellBox(CellHouse):
    @property
    def house_type(self) -> str:
        return "box"

    @property
    def name(self) -> str:
        return f"Box {self.index + 1}"
