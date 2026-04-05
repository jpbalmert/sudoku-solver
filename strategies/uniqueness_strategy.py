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
       because its value is already solved in one of that cell's houses.
    """

    @property
    def name(self) -> str:
        return "Uniqueness"

    def apply(self, game: Game) -> StepResult | None:
        if not game.has_candidates():
            return self._initialize_candidates(game)

        # Populate candidates for any empty cells that are missing them
        # (e.g. cells the OCR couldn't read candidates for)
        result = self._populate_missing_candidates(game)
        if result:
            return result

        # Look for a solved cell whose value can be eliminated from peers
        return self._eliminate_from_source(game)

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

    def _populate_missing_candidates(self, game: Game) -> StepResult | None:
        """Populate candidates for empty cells that have none yet."""
        all_digits = set(range(1, 10))
        changed: list[Cell] = []

        for row in game.grid:
            for cell in row:
                if cell.is_empty and not cell.candidates:
                    houses = game.houses_for_cell(cell)
                    used = set()
                    for house in houses:
                        used |= house.solved_values
                    cell.candidates = all_digits - used
                    changed.append(cell)

        if not changed:
            return None

        return StepResult(
            description=(
                "Populated candidate values for empty cells by removing "
                "digits already present in each cell's row, column, and box."
            ),
            changed_cells=changed,
            is_initialization=True,
        )

    def _eliminate_from_source(self, game: Game) -> StepResult | None:
        """Find a solved cell and eliminate its value from all peers at once.

        Scans every solved cell. For the first one whose value still appears
        as a candidate in any peer cell (same row, column, or box), removes
        that value from ALL such peers in a single step.
        """
        for row in game.grid:
            for source in row:
                if source.value is None:
                    continue
                value = source.value

                # Collect all peer cells that still have this value as a candidate
                affected: list[Cell] = []
                for house in game.houses_for_cell(source):
                    for cell in house.cells:
                        if cell is not source and cell.is_empty and value in cell.candidates:
                            if cell not in affected:
                                affected.append(cell)

                if not affected:
                    continue

                # Remove the candidate from all affected cells
                eliminated: dict[tuple[int, int], set[int]] = {}
                for cell in affected:
                    cell.candidates.discard(value)
                    eliminated[(cell.row, cell.col)] = {value}

                # Build description
                cell_refs = ", ".join(
                    f"R{c.row + 1}C{c.col + 1}" for c in affected
                )
                return StepResult(
                    description=(
                        f"Removed candidate {value} from {cell_refs} "
                        f"because {value} is already placed at "
                        f"R{source.row + 1}C{source.col + 1}."
                    ),
                    changed_cells=affected,
                    contributing_cells=[source],
                    eliminated_candidates=eliminated,
                )
        return None

