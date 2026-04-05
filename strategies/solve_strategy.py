from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from models.cell import Cell


@dataclass
class StepResult:
    """The outcome of applying a single strategy step."""

    description: str
    changed_cells: list[Cell] = field(default_factory=list)
    contributing_cells: list[Cell] = field(default_factory=list)
    eliminated_candidates: dict[tuple[int, int], set[int]] = field(default_factory=dict)
    solved_cell: tuple[int, int] | None = None
    solved_value: int | None = None
    is_initialization: bool = False


class SolveStrategy(ABC):
    """Interface for a Sudoku solving strategy."""

    @abstractmethod
    def apply(self, game) -> StepResult | None:
        """Apply the strategy to the game.

        Returns a StepResult if the strategy made a change, or None if it
        found nothing to do.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
