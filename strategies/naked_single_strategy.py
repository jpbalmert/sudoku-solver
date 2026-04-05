from models.game import Game
from .solve_strategy import SolveStrategy, StepResult


class NakedSingleStrategy(SolveStrategy):
    """Solves a cell that has exactly one remaining candidate.

    When all but one candidate have been eliminated from a cell, the
    remaining candidate must be the cell's value.
    """

    @property
    def name(self) -> str:
        return "Naked Single"

    def apply(self, game: Game) -> StepResult | None:
        for row in game.grid:
            for cell in row:
                if cell.is_empty and len(cell.candidates) == 1:
                    value = next(iter(cell.candidates))
                    cell.candidates.clear()
                    cell.solved = value
                    return StepResult(
                        description=(
                            f"Solved R{cell.row + 1}C{cell.col + 1} = {value}. "
                            f"It was the only remaining candidate in this cell."
                        ),
                        changed_cells=[cell],
                        solved_cell=(cell.row, cell.col),
                        solved_value=value,
                    )
        return None
