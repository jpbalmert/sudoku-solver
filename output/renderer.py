from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from models.game import Game
from models.cell import Cell
from strategies.solve_strategy import StepResult
from validation.validator import ValidationError

# Grid layout constants
IMAGE_SIZE = 900
GRID_ORIGIN = 45  # padding around the grid
GRID_SIZE = IMAGE_SIZE - 2 * GRID_ORIGIN  # 810
CELL_SIZE = GRID_SIZE // 9  # 90
THIN_LINE = 1
THICK_LINE = 3

# Colors
COLOR_BG = (255, 255, 255)
COLOR_GRID_LINE = (0, 0, 0)
COLOR_GIVEN = (0, 0, 0)
COLOR_SOLVED = (0, 0, 200)
COLOR_CANDIDATE = (0, 0, 200)
COLOR_CONFLICT = (220, 0, 0)
COLOR_CHANGED_BG = (255, 255, 200)
COLOR_CONTRIBUTING_BG = (220, 220, 220)
COLOR_STRIKETHROUGH = (220, 0, 0)


def _get_fonts() -> tuple:
    """Load fonts, falling back to default if needed."""
    try:
        large_font = ImageFont.truetype("arial.ttf", 48)
        small_font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        try:
            large_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except OSError:
            large_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
    return large_font, small_font


def _cell_rect(row: int, col: int) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) pixel coordinates for a cell."""
    x0 = GRID_ORIGIN + col * CELL_SIZE
    y0 = GRID_ORIGIN + row * CELL_SIZE
    return x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE


def render_image(
    game: Game,
    step_result: StepResult | None = None,
    validation_errors: list[ValidationError] | None = None,
    output_path: str | Path = "output.png",
) -> None:
    """Render the current game state to a PNG image."""
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), COLOR_BG)
    draw = ImageDraw.Draw(img)
    large_font, small_font = _get_fonts()

    # Collect cells needing special backgrounds
    changed_cells: set[tuple[int, int]] = set()
    contributing_cells: set[tuple[int, int]] = set()
    conflict_cells: set[tuple[int, int]] = set()
    eliminated: dict[tuple[int, int], set[int]] = {}

    if step_result:
        for c in step_result.changed_cells:
            changed_cells.add((c.row, c.col))
        for c in step_result.contributing_cells:
            contributing_cells.add((c.row, c.col))
        eliminated = step_result.eliminated_candidates

    if validation_errors:
        for err in validation_errors:
            for c in err.conflicting_cells:
                conflict_cells.add((c.row, c.col))

    # Draw cell backgrounds
    for r in range(9):
        for c in range(9):
            key = (r, c)
            x0, y0, x1, y1 = _cell_rect(r, c)
            if key in conflict_cells:
                pass  # no special background for conflicts, digits turn red
            elif key in changed_cells and not (step_result and step_result.is_initialization):
                draw.rectangle([x0, y0, x1, y1], fill=COLOR_CHANGED_BG)
            elif key in contributing_cells:
                draw.rectangle([x0, y0, x1, y1], fill=COLOR_CONTRIBUTING_BG)

    # Draw cell contents
    for r in range(9):
        for c in range(9):
            cell = game.cell(r, c)
            x0, y0, x1, y1 = _cell_rect(r, c)
            is_conflict = (r, c) in conflict_cells

            if cell.given is not None:
                _draw_large_digit(
                    draw, cell.given, x0, y0,
                    COLOR_CONFLICT if is_conflict else COLOR_GIVEN,
                    large_font,
                )
            elif cell.solved is not None:
                _draw_large_digit(
                    draw, cell.solved, x0, y0,
                    COLOR_CONFLICT if is_conflict else COLOR_SOLVED,
                    large_font,
                )
            elif cell.candidates:
                elim = eliminated.get((r, c), set())
                _draw_candidates(draw, cell.candidates, elim, x0, y0, small_font)

    # Draw grid lines
    _draw_grid(draw)

    img.save(str(output_path))


def _draw_large_digit(
    draw: ImageDraw.ImageDraw,
    digit: int,
    x0: int, y0: int,
    color: tuple,
    font: ImageFont.FreeTypeFont,
) -> None:
    """Draw a full-size digit centered in the cell."""
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = x0 + (CELL_SIZE - tw) // 2
    y = y0 + (CELL_SIZE - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=color, font=font)


def _draw_candidates(
    draw: ImageDraw.ImageDraw,
    candidates: set[int],
    eliminated: set[int],
    x0: int, y0: int,
    font: ImageFont.FreeTypeFont,
) -> None:
    """Draw small candidate digits in a 3x3 layout within the cell."""
    sub_w = CELL_SIZE // 3
    sub_h = CELL_SIZE // 3

    all_to_draw = candidates | eliminated

    for digit in all_to_draw:
        # Position: 1-3 in row 0, 4-6 in row 1, 7-9 in row 2
        dr = (digit - 1) // 3
        dc = (digit - 1) % 3
        sx = x0 + dc * sub_w
        sy = y0 + dr * sub_h

        text = str(digit)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = sx + (sub_w - tw) // 2
        ty = sy + (sub_h - th) // 2 - bbox[1]

        if digit in eliminated:
            draw.text((tx, ty), text, fill=COLOR_STRIKETHROUGH, font=font)
            # Draw strikethrough line
            line_y = ty + th // 2 + bbox[1]
            draw.line(
                [(tx - 1, line_y), (tx + tw + 1, line_y)],
                fill=COLOR_STRIKETHROUGH,
                width=2,
            )
        else:
            draw.text((tx, ty), text, fill=COLOR_CANDIDATE, font=font)


def _draw_grid(draw: ImageDraw.ImageDraw) -> None:
    """Draw the Sudoku grid lines."""
    for i in range(10):
        width = THICK_LINE if i % 3 == 0 else THIN_LINE
        # Horizontal
        y = GRID_ORIGIN + i * CELL_SIZE
        draw.line(
            [(GRID_ORIGIN, y), (GRID_ORIGIN + GRID_SIZE, y)],
            fill=COLOR_GRID_LINE,
            width=width,
        )
        # Vertical
        x = GRID_ORIGIN + i * CELL_SIZE
        draw.line(
            [(x, GRID_ORIGIN), (x, GRID_ORIGIN + GRID_SIZE)],
            fill=COLOR_GRID_LINE,
            width=width,
        )


def render_text(
    step_result: StepResult | None = None,
    validation_errors: list[ValidationError] | None = None,
    output_path: str | Path = "output.txt",
) -> None:
    """Write the reasoning text for this step to a file."""
    lines: list[str] = []

    if validation_errors:
        lines.append("VALIDATION ERRORS:")
        for err in validation_errors:
            lines.append(f"  - {err.description}")
        lines.append("")

    if step_result:
        lines.append(f"Strategy: {step_result.description}")
    else:
        lines.append("Unable to proceed.")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
