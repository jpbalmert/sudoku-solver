from models.game import Game
from models.cell import Cell
from .solve_strategy import SolveStrategy, StepResult


class UniquenessStrategy(SolveStrategy):
    """Applies the basic Sudoku uniqueness constraint.

    Each digit 1-9 must appear exactly once in every row, column, and box.
    This strategy:
    1. On a fresh puzzle (no candidates): populates all candidates for every
       empty cell based on what values are already present in its houses.
    2. On a puzzle with candidates: finds one candidate that can be eliminated
       because its value is already solved in one of that cell's houses, OR
       solves a cell that has exactly one remaining candidate.
    """

    @property
    def name(self) -> str:
        return "Uniqueness"

    def apply(self, game: Game) -> StepResult | None:
        if not game.has_candidates():
            return self._initialize_candidates(game)

        # Look for a candidate to eliminate
        result = self._eliminate_one(game)
        if result:
            return result

        # Look for a naked single to solve
        return self._solve_naked_single(game)

    def _initialize_candidates(self, game: Game) -> StepResult:
        """Populate candidates for all empty cells."""
        all_digits = set(range(1, 10))
        changed: list[Cell] = []

        for row in game.grid:
            for cell in row:
                if cell.is_empty:
                    houses = game.houses_for_cell(cell)
                    used = set()
                    for house in houses:
                        used |= house.solved_values
                    cell.candidates = all_digits - used
                    changed.append(cell)

        return StepResult(
            description=(
                "Populated candidate values for all empty cells by removing "
                "digits already present in each cell's row, column, and box."
            ),
            changed_cells=changed,
            is_initialization=True,
        )

    def _eliminate_one(self, game: Game) -> StepResult | None:
        """Find one candidate that conflicts with a solved value in its house."""
        for row in game.grid:
            for cell in row:
                if not cell.is_empty or not cell.candidates:
                    continue
                for house in game.houses_for_cell(cell):
                    conflicts = cell.candidates & house.solved_values
                    if conflicts:
                        value = min(conflicts)
                        # Find the solved cell that causes the conflict
                        source = next(
                            c for c in house.cells
                            if c.value == value and c is not cell
                        )
                        cell.candidates.discard(value)
                        return StepResult(
                            description=(
                                f"Removed candidate {value} from "
                                f"R{cell.row + 1}C{cell.col + 1} because "
                                f"{value} is already placed in {house.name}."
                            ),
                            changed_cells=[cell],
                            contributing_cells=[source],
                            eliminated_candidates={
                                (cell.row, cell.col): {value}
                            },
                        )
        return None

    def _solve_naked_single(self, game: Game) -> StepResult | None:
        """Solve a cell that has exactly one remaining candidate."""
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
