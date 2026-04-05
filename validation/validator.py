from dataclasses import dataclass
from models.game import Game
from models.cell import Cell


@dataclass
class ValidationError:
    """A conflict found during validation."""

    description: str
    conflicting_cells: list[Cell]


def validate(game: Game) -> list[ValidationError]:
    """Check the game state for Sudoku rule violations.

    Returns a list of ValidationError objects. An empty list means the state
    is valid.
    """
    errors: list[ValidationError] = []

    for house in game.all_houses():
        seen: dict[int, Cell] = {}
        for cell in house.cells:
            v = cell.value
            if v is None:
                continue
            if v in seen:
                errors.append(ValidationError(
                    description=(
                        f"Duplicate {v} in {house.name}: "
                        f"R{seen[v].row + 1}C{seen[v].col + 1} and "
                        f"R{cell.row + 1}C{cell.col + 1}"
                    ),
                    conflicting_cells=[seen[v], cell],
                ))
            else:
                seen[v] = cell

    return errors
