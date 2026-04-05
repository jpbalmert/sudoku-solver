import cv2
import numpy as np
import pytesseract
from pathlib import Path
from models.game import Game


# Supported image formats (add new extensions here to support more formats)
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg"}


def ingest_image(path: str | Path) -> Game:
    """Read a Sudoku puzzle image and build a Game model from it.

    Args:
        path: Path to a PNG or JPEG image of a Sudoku puzzle.

    Returns:
        A Game with givens, solved cells, and candidates populated based on
        what was detected in the image.
    """
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # TODO: Inject perspective correction logic here for skewed/rotated images.
    # For now we assume the grid is roughly axis-aligned and fills most of
    # the frame. A future implementation could use cv2.findContours and
    # cv2.getPerspectiveTransform to straighten the grid.

    grid_img = _extract_grid(img)
    game = Game()
    _populate_cells(game, grid_img)
    return game


def _extract_grid(img: np.ndarray) -> np.ndarray:
    """Locate and extract the Sudoku grid region from the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2,
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError("Could not find any contours in the image.")

    # Find the largest contour by area — should be the outer grid boundary
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Extract and resize to a standard size for consistent cell extraction
    grid = img[y:y + h, x:x + w]
    grid = cv2.resize(grid, (810, 810))
    return grid


def _populate_cells(game: Game, grid_img: np.ndarray) -> None:
    """Extract digits and candidates from each cell in the grid image."""
    cell_size = 90  # 810 / 9

    for r in range(9):
        for c in range(9):
            x0 = c * cell_size
            y0 = r * cell_size
            cell_img = grid_img[y0:y0 + cell_size, x0:x0 + cell_size]
            _process_cell(game.cell(r, c), cell_img)


def _process_cell(cell, cell_img: np.ndarray) -> None:
    """Determine the contents of a single cell from its image."""
    # Crop a small margin to avoid grid lines
    margin = 8
    interior = cell_img[margin:cell_img.shape[0] - margin,
                        margin:cell_img.shape[1] - margin]

    gray = cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Count non-zero pixels to detect if the cell has content
    pixel_count = cv2.countNonZero(binary)
    total_pixels = binary.shape[0] * binary.shape[1]

    if pixel_count < total_pixels * 0.02:
        # Cell is empty
        return

    # Detect if this is a large digit or small candidates
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return

    # Find the bounding box of all content
    all_points = np.vstack(contours)
    bx, by, bw, bh = cv2.boundingRect(all_points)

    # Heuristic: if the content bounding box is large relative to the cell,
    # it's a single digit. If it's spread out with multiple small clusters,
    # it's candidates.
    content_ratio = (bw * bh) / (binary.shape[0] * binary.shape[1])
    num_clusters = len([ct for ct in contours if cv2.contourArea(ct) > 15])

    if num_clusters <= 2 and content_ratio < 0.6:
        # Likely a single large digit
        digit, is_blue = _ocr_digit(interior, cell_img)
        if digit:
            if is_blue:
                cell.solved = digit
            else:
                cell.given = digit
    else:
        # Likely candidates — read each position in the 3x3 mini-grid
        _ocr_candidates(cell, interior, cell_img)


def _ocr_digit(interior: np.ndarray, color_img: np.ndarray) -> tuple[int | None, bool]:
    """OCR a single large digit from a cell interior.

    Returns (digit, is_blue) where is_blue indicates if the digit is blue
    (solved) vs black (given).
    """
    # Prepare for Tesseract: enlarge for better recognition
    scaled = cv2.resize(interior, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(
        binary,
        config="--psm 10 -c tessedit_char_whitelist=123456789",
    ).strip()

    digit = None
    if text and text[0].isdigit():
        digit = int(text[0])

    is_blue = _is_blue_digit(color_img)
    return digit, is_blue


def _ocr_candidates(cell, interior: np.ndarray, color_img: np.ndarray) -> None:
    """OCR candidate digits from the 3x3 mini-grid positions within a cell."""
    h, w = interior.shape[:2]
    sub_h = h // 3
    sub_w = w // 3

    for digit in range(1, 10):
        dr = (digit - 1) // 3
        dc = (digit - 1) % 3
        sx = dc * sub_w
        sy = dr * sub_h
        sub_img = interior[sy:sy + sub_h, sx:sx + sub_w]

        gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        pixel_count = cv2.countNonZero(binary)

        if pixel_count < (sub_h * sub_w) * 0.03:
            continue

        # Enlarge for OCR
        scaled = cv2.resize(sub_img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray_scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        _, bin_scaled = cv2.threshold(gray_scaled, 180, 255, cv2.THRESH_BINARY)

        text = pytesseract.image_to_string(
            bin_scaled,
            config="--psm 10 -c tessedit_char_whitelist=123456789",
        ).strip()

        if text and text[0].isdigit():
            cell.candidates.add(int(text[0]))


def _is_blue_digit(cell_img: np.ndarray) -> bool:
    """Determine if the digit in the cell image is blue (solved) vs black (given)."""
    # Convert to HSV for easier color detection
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)

    # Blue range in HSV
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Count blue pixels — if significant, it's a blue digit
    blue_count = cv2.countNonZero(blue_mask)
    total = cell_img.shape[0] * cell_img.shape[1]

    return blue_count > total * 0.01
