import cv2
import numpy as np
import pytesseract
from pathlib import Path
from models.game import Game


# Supported image formats (add new extensions here to support more formats)
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg"}

# Minimum height ratio of content bounding box vs cell interior to count as
# a large (given/solved) digit.  Candidates are much smaller.
LARGE_DIGIT_HEIGHT_RATIO = 0.45


def ingest_image(path: str | Path) -> Game:
    """Read a Sudoku puzzle image and build a Game model from it.

    Ingestion is done in two passes:
      Pass 1 — Read large digits (givens and solved values).
      Pass 2 — Read small candidate digits in cells that have no large digit.

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

    grid_img, h_lines, v_lines = _extract_grid(img)
    cell_images = _extract_cell_images(grid_img, h_lines, v_lines)
    game = Game()

    # Pass 1: Read large digits only
    for r in range(9):
        for c in range(9):
            _read_large_digit(game.cell(r, c), cell_images[r][c])

    # Pass 2: Read candidates only (skip cells that already have a value)
    for r in range(9):
        for c in range(9):
            cell = game.cell(r, c)
            if cell.value is None:
                _read_candidates(cell, cell_images[r][c])

    return game


# ---------------------------------------------------------------------------
# Grid extraction
# ---------------------------------------------------------------------------

def _extract_grid(img: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    """Locate the Sudoku grid and find the 10 horizontal and 10 vertical lines.

    Returns:
        (grid_image, h_lines, v_lines) where h_lines and v_lines are lists of
        10 pixel positions for the horizontal and vertical grid lines within
        the returned grid image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2,
    )

    # Find the outer grid boundary
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError("Could not find any contours in the image.")

    largest = max(contours, key=cv2.contourArea)
    gx, gy, gw, gh = cv2.boundingRect(largest)

    # Crop to grid region
    grid_img = img[gy:gy + gh, gx:gx + gw]
    grid_thresh = thresh[gy:gy + gh, gx:gx + gw]

    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gw // 3, 1))
    h_lines_img = cv2.morphologyEx(grid_thresh, cv2.MORPH_OPEN, h_kernel)
    h_positions = _find_line_positions(h_lines_img, axis=0)

    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gh // 3))
    v_lines_img = cv2.morphologyEx(grid_thresh, cv2.MORPH_OPEN, v_kernel)
    v_positions = _find_line_positions(v_lines_img, axis=1)

    # We expect 10 lines in each direction. If we don't get exactly 10,
    # fall back to evenly spaced lines.
    if len(h_positions) != 10:
        h_positions = [int(i * gh / 9) for i in range(10)]
    if len(v_positions) != 10:
        v_positions = [int(i * gw / 9) for i in range(10)]

    return grid_img, h_positions, v_positions


def _find_line_positions(line_img: np.ndarray, axis: int) -> list[int]:
    """Find the positions of lines by projecting and finding peaks.

    Args:
        line_img: Binary image with only horizontal or vertical lines.
        axis: 0 for horizontal lines (project onto Y axis),
              1 for vertical lines (project onto X axis).
    """
    if axis == 0:
        projection = np.sum(line_img, axis=1)
    else:
        projection = np.sum(line_img, axis=0)

    projection = projection.astype(float)
    if projection.max() == 0:
        return []

    threshold = projection.max() * 0.3
    above = projection > threshold

    positions = []
    in_run = False
    start = 0
    for i, val in enumerate(above):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            positions.append((start + i) // 2)
            in_run = False
    if in_run:
        positions.append((start + len(above)) // 2)

    return positions


def _extract_cell_images(
    grid_img: np.ndarray,
    h_lines: list[int],
    v_lines: list[int],
) -> list[list[np.ndarray]]:
    """Cut the grid into 9x9 cell images with margins trimmed."""
    cells = []
    for r in range(9):
        row = []
        for c in range(9):
            y0, y1 = h_lines[r], h_lines[r + 1]
            x0, x1 = v_lines[c], v_lines[c + 1]
            row.append(grid_img[y0:y1, x0:x1])
        cells.append(row)
    return cells


def _get_interior(cell_img: np.ndarray) -> np.ndarray:
    """Crop margins from a cell image to avoid grid lines."""
    h, w = cell_img.shape[:2]
    margin_x = max(3, w // 12)
    margin_y = max(3, h // 12)
    return cell_img[margin_y:h - margin_y, margin_x:w - margin_x]


# ---------------------------------------------------------------------------
# Pass 1: Large digits
# ---------------------------------------------------------------------------

def _read_large_digit(cell, cell_img: np.ndarray) -> None:
    """Attempt to read a single large digit from the cell.

    Uses a simple global threshold (black text on white) and tries multiple
    threshold values and Tesseract PSM modes.
    """
    interior = _get_interior(cell_img)
    if interior.shape[0] < 10 or interior.shape[1] < 10:
        return

    gray = cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY)

    # Check if there is a tall enough contour to be a large digit.
    # Use adaptive threshold for content detection (handles all ink colors).
    detect_binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    detect_binary = cv2.morphologyEx(detect_binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        detect_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return

    max_height = max(cv2.boundingRect(ct)[3] for ct in contours)
    cell_h = detect_binary.shape[0]

    if max_height < cell_h * LARGE_DIGIT_HEIGHT_RATIO:
        return  # Not a large digit — will be handled in pass 2

    # OCR: try multiple thresholds with black-on-white for Tesseract
    digit = _ocr_large_digit(gray)
    if digit:
        is_blue = _is_blue_digit(cell_img)
        if is_blue:
            cell.solved = digit
        else:
            cell.given = digit


def _ocr_large_digit(gray: np.ndarray) -> int | None:
    """OCR a single large digit from a grayscale cell interior.

    Tries multiple threshold values and PSM modes to maximize recognition.
    """
    for thresh_val in [220, 200, 180, 160, 140]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        scaled = cv2.resize(binary, (0, 0), fx=3, fy=3,
                            interpolation=cv2.INTER_CUBIC)
        for psm in [8, 6]:
            text = pytesseract.image_to_string(
                scaled,
                config=f"--psm {psm} -c tessedit_char_whitelist=123456789",
            ).strip()
            if text and text[0].isdigit():
                return int(text[0])
    return None


def _is_blue_digit(cell_img: np.ndarray) -> bool:
    """Determine if the digit in the cell image is blue (solved) vs black (given)."""
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)

    # Blue range in HSV
    lower_blue = np.array([90, 30, 30])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find pixels that are part of the digit (dark enough)
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    _, digit_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    digit_pixel_count = cv2.countNonZero(digit_mask)

    if digit_pixel_count == 0:
        return False

    blue_digit = cv2.bitwise_and(blue_mask, digit_mask)
    blue_digit_count = cv2.countNonZero(blue_digit)

    return blue_digit_count > digit_pixel_count * 0.70


# ---------------------------------------------------------------------------
# Pass 2: Candidates
# ---------------------------------------------------------------------------

def _read_candidates(cell, cell_img: np.ndarray) -> None:
    """Read small candidate digits from the 3x3 mini-grid positions.

    Since candidates are in fixed positions (1-3 top row, 4-6 middle, 7-9
    bottom), we check each sub-region for content and assign the digit by
    its position rather than relying on OCR for tiny text.
    """
    interior = _get_interior(cell_img)
    if interior.shape[0] < 10 or interior.shape[1] < 10:
        return

    h, w = interior.shape[:2]
    sub_h = h // 3
    sub_w = w // 3

    for digit in range(1, 10):
        dr = (digit - 1) // 3
        dc = (digit - 1) % 3
        sx = dc * sub_w
        sy = dr * sub_h
        sub_img = interior[sy:sy + sub_h, sx:sx + sub_w]

        if _sub_region_has_content(sub_img):
            cell.candidates.add(digit)


def _sub_region_has_content(sub_img: np.ndarray) -> bool:
    """Detect whether a candidate sub-region contains a small digit.

    Uses adaptive thresholding to handle black, blue, and grey text.
    Red pixels are masked out first so that struck-through (eliminated)
    candidates from a previous step's output are not re-read.
    """
    # Mask out red pixels before content detection
    filtered = _mask_red_pixels(sub_img)

    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    pixel_count = cv2.countNonZero(binary)
    total = sub_img.shape[0] * sub_img.shape[1]

    return pixel_count > total * 0.04


def _mask_red_pixels(img: np.ndarray) -> np.ndarray:
    """Replace red pixels with white so they are ignored during detection."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red wraps around in HSV, so we need two ranges
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    result = img.copy()
    result[mask > 0] = (255, 255, 255)
    return result
