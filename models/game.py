from .cell import Cell
from .cell_house import CellHouse, CellRow, CellColumn, CellBox


class Game:
    """The complete state of a Sudoku puzzle."""

    def __init__(self):
        self.grid: list[list[Cell]] = [
            [Cell(r, c) for c in range(9)] for r in range(9)
        ]

        self.rows: list[CellRow] = [
            CellRow(r, self.grid[r]) for r in range(9)
        ]

        self.columns: list[CellColumn] = [
            CellColumn(c, [self.grid[r][c] for r in range(9)]) for c in range(9)
        ]

        self.boxes: list[CellBox] = []
        for box_idx in range(9):
            br = (box_idx // 3) * 3
            bc = (box_idx % 3) * 3
            cells = [
                self.grid[br + r][bc + c]
                for r in range(3)
                for c in range(3)
            ]
            self.boxes.append(CellBox(box_idx, cells))

    def cell(self, row: int, col: int) -> Cell:
        return self.grid[row][col]

    def houses_for_cell(self, cell: Cell) -> list[CellHouse]:
        """Return the row, column, and box that contain the given cell."""
        return [
            self.rows[cell.row],
            self.columns[cell.col],
            self.boxes[cell.box_index],
        ]

    def all_houses(self) -> list[CellHouse]:
        """Return all 27 houses (9 rows + 9 columns + 9 boxes)."""
        return self.rows + self.columns + self.boxes

    def has_candidates(self) -> bool:
        """True if any cell in the grid has candidates populated."""
        return any(
            cell.candidates
            for row in self.grid
            for cell in row
        )

    def is_solved(self) -> bool:
        """True if every cell has a value."""
        return all(
            cell.value is not None
            for row in self.grid
            for cell in row
        )
