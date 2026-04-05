from models.game import Game
from .solve_strategy import SolveStrategy, StepResult


class HiddenSingleStrategy(SolveStrategy):
    """Solves a cell where a candidate appears in only one cell within a house.

    If a digit can only go in one cell within a row, column, or box, then
    that cell must contain that digit — even if the cell has other candidates.
    """

    @property
    def name(self) -> str:
        return "Hidden Single"

    def apply(self, game: Game) -> StepResult | None:
        for house in game.all_houses():
            for digit in range(1, 10):
                if digit in house.solved_values:
                    continue

                # Find cells in this house that have this digit as a candidate
                possible_cells = [
                    c for c in house.cells
                    if c.is_empty and digit in c.candidates
                ]

                if len(possible_cells) == 1:
                    cell = possible_cells[0]
                    # All other cells in the house that already have this digit
                    # excluded are the contributing cells (they forced it here)
                    contributing = [
                        c for c in house.cells
                        if c.value is not None and c is not cell
                    ]
                    cell.candidates.clear()
                    cell.solved = digit
                    return StepResult(
                        description=(
                            f"Solved R{cell.row + 1}C{cell.col + 1} = {digit}. "
                            f"It was the only cell in {house.name} that could "
                            f"contain {digit}."
                        ),
                        changed_cells=[cell],
                        contributing_cells=contributing,
                        solved_cell=(cell.row, cell.col),
                        solved_value=digit,
                    )
        return None
