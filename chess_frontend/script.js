/*
 * Chess FEN Trainer
 *
 * This script builds a chessboard from a Forsyth‑Edwards Notation (FEN)
 * string, allows the user to make the next move via click interactions and
 * compares the move against a pseudo‑random suggestion from a dummy
 * Large‑Language‑Model (LLM). The move generation here is "pseudo legal": it
 * follows the basic movement rules for each piece but does not check for
 * check, castling or en‑passant. That level of complexity isn’t required
 * for the purposes of comparing moves. See the FEN specification for
 * details on the notation【115677098837149†L133-L165】.
 */

// Mapping from FEN piece characters to Unicode symbols for display.
const PIECE_UNICODE = {
  p: '♟',
  n: '♞',
  b: '♝',
  r: '♜',
  q: '♛',
  k: '♚',
  P: '♙',
  N: '♘',
  B: '♗',
  R: '♖',
  Q: '♕',
  K: '♔'
};

// Board state: a 2D array [row][col] where each element is either null
// (empty square) or an object with properties {type, color, char}.
let boardState = [];

// Whose turn it is: 'w' for white, 'b' for black.
let activeColor = 'w';

// Whether the user has made a move; after one move the board is frozen.
let userMoved = false;

// The LLM’s suggested move, generated when a position is loaded. It has
// properties {from: 'e2', to: 'e4'}.
let llmMove = null;

// Index of the current entry from the CSV dataset. If ENTRIES is defined
// (imported via data.js), the app cycles through these records. Otherwise
// the user can manually enter FEN strings. 0-based index.
let currentEntryIndex = 0;

// Internal state for the currently selected square and its legal moves.
let selectedSquare = null;
let legalMoves = [];

// DOM elements for convenience
const fenInput = document.getElementById('fen-input');
const loadButton = document.getElementById('load-button');
const boardElement = document.getElementById('board');
const instructionElement = document.getElementById('instruction');
const statusElement = document.getElementById('status');
const llmElement = document.getElementById('llm-move');


// Navigation DOM elements (may not exist in older versions)
const navigationElement = document.getElementById('navigation');
const nextButton = document.getElementById('next-button');

/**
 * Parse a FEN string into board state and active colour. Only the first two
 * fields (piece placement and active colour) are required for this app.
 *
 * @param {string} fen The FEN string entered by the user.
 * @returns {boolean} True if the FEN was parsed successfully, false otherwise.
 */
function parseFEN(fen) {
  const parts = fen.trim().split(/\s+/);
  if (parts.length < 2) return false;
  const placement = parts[0];
  const active = parts[1];
  // Validate the active colour field
  if (active !== 'w' && active !== 'b') return false;
  activeColor = active;
  // Reset board state
  boardState = new Array(8).fill(null).map(() => new Array(8).fill(null));
  const ranks = placement.split('/');
  if (ranks.length !== 8) return false;
  for (let r = 0; r < 8; r++) {
    const rankStr = ranks[r];
    let file = 0;
    for (let i = 0; i < rankStr.length; i++) {
      const ch = rankStr[i];
      if (/[1-8]/.test(ch)) {
        const empty = parseInt(ch, 10);
        file += empty;
      } else if (/[pnbrqkPNBRQK]/.test(ch)) {
        const color = ch === ch.toLowerCase() ? 'b' : 'w';
        boardState[r][file] = {
          type: ch.toLowerCase(),
          color: color,
          char: ch
        };
        file++;
      } else {
        // Invalid character
        return false;
      }
    }
    if (file !== 8) return false;
  }
  return true;
}

/**
 * Convert board coordinates to algebraic notation.
 *
 * @param {number} row Row index (0 at top)
 * @param {number} col Column index (0 at left)
 * @returns {string} Algebraic square (e.g. 'e4')
 */
function coordToSquare(row, col) {
  const file = String.fromCharCode('a'.charCodeAt(0) + col);
  const rank = 8 - row;
  return file + rank;
}

/**
 * Convert algebraic notation to board coordinates.
 *
 * @param {string} square Algebraic square (e.g. 'e4')
 * @returns {{row: number, col: number}} row and col indices
 */
function squareToCoord(square) {
  const fileChar = square[0];
  const rankChar = square[1];
  const col = fileChar.charCodeAt(0) - 'a'.charCodeAt(0);
  const row = 8 - parseInt(rankChar, 10);
  return { row, col };
}

/**
 * Generate pseudo‑legal moves for a pawn.
 *
 * @param {number} row Starting row
 * @param {number} col Starting column
 * @param {string} color 'w' or 'b'
 * @returns {Array<{from: {row:number,col:number}, to:{row:number,col:number}}>} List of moves
 */
function generatePawnMoves(row, col, color) {
  const moves = [];
  const dir = color === 'w' ? -1 : 1;
  const startRow = color === 'w' ? 6 : 1;
  // Forward one
  const forwardRow = row + dir;
  if (forwardRow >= 0 && forwardRow < 8) {
    if (!boardState[forwardRow][col]) {
      moves.push({ from: { row, col }, to: { row: forwardRow, col } });
      // Forward two from starting rank
      const forwardTwoRow = row + 2 * dir;
      if (row === startRow && !boardState[forwardTwoRow][col]) {
        moves.push({ from: { row, col }, to: { row: forwardTwoRow, col } });
      }
    }
    // Captures
    for (const dc of [-1, 1]) {
      const captureCol = col + dc;
      if (captureCol >= 0 && captureCol < 8) {
        const target = boardState[forwardRow][captureCol];
        if (target && target.color !== color) {
          moves.push({ from: { row, col }, to: { row: forwardRow, col: captureCol } });
        }
      }
    }
  }
  return moves;
}

/**
 * Generate pseudo‑legal moves for a knight.
 */
function generateKnightMoves(row, col, color) {
  const moves = [];
  const deltas = [
    [-2, -1], [-2, 1],
    [-1, -2], [-1, 2],
    [1, -2], [1, 2],
    [2, -1], [2, 1]
  ];
  for (const [dr, dc] of deltas) {
    const r = row + dr;
    const c = col + dc;
    if (r >= 0 && r < 8 && c >= 0 && c < 8) {
      const target = boardState[r][c];
      if (!target || target.color !== color) {
        moves.push({ from: { row, col }, to: { row: r, col: c } });
      }
    }
  }
  return moves;
}

/**
 * Generate sliding moves in given directions until blocked.
 */
function generateSlidingMoves(row, col, color, directions) {
  const moves = [];
  for (const [dr, dc] of directions) {
    let r = row + dr;
    let c = col + dc;
    while (r >= 0 && r < 8 && c >= 0 && c < 8) {
      const target = boardState[r][c];
      if (!target) {
        moves.push({ from: { row, col }, to: { row: r, col: c } });
      } else {
        if (target.color !== color) {
          moves.push({ from: { row, col }, to: { row: r, col: c } });
        }
        break;
      }
      r += dr;
      c += dc;
    }
  }
  return moves;
}

/**
 * Generate pseudo‑legal moves for a bishop.
 */
function generateBishopMoves(row, col, color) {
  return generateSlidingMoves(row, col, color, [[-1, -1], [-1, 1], [1, -1], [1, 1]]);
}

/**
 * Generate pseudo‑legal moves for a rook.
 */
function generateRookMoves(row, col, color) {
  return generateSlidingMoves(row, col, color, [[-1, 0], [1, 0], [0, -1], [0, 1]]);
}

/**
 * Generate pseudo‑legal moves for a queen.
 */
function generateQueenMoves(row, col, color) {
  return generateSlidingMoves(row, col, color, [
    [-1, -1], [-1, 1], [1, -1], [1, 1],
    [-1, 0], [1, 0], [0, -1], [0, 1]
  ]);
}

/**
 * Generate pseudo‑legal moves for a king (no castling considered).
 */
function generateKingMoves(row, col, color) {
  const moves = [];
  for (const dr of [-1, 0, 1]) {
    for (const dc of [-1, 0, 1]) {
      if (dr === 0 && dc === 0) continue;
      const r = row + dr;
      const c = col + dc;
      if (r >= 0 && r < 8 && c >= 0 && c < 8) {
        const target = boardState[r][c];
        if (!target || target.color !== color) {
          moves.push({ from: { row, col }, to: { row: r, col: c } });
        }
      }
    }
  }
  return moves;
}

/**
 * Generate pseudo‑legal moves for a particular piece on the board. Castling
 * and en‑passant are intentionally omitted for simplicity.
 *
 * @param {number} row The row index of the piece
 * @param {number} col The column index of the piece
 * @param {Object} piece The piece object with `type` and `color` fields
 * @returns {Array} List of move objects
 */
function generateMovesForPiece(row, col, piece) {
  switch (piece.type) {
    case 'p':
      return generatePawnMoves(row, col, piece.color);
    case 'n':
      return generateKnightMoves(row, col, piece.color);
    case 'b':
      return generateBishopMoves(row, col, piece.color);
    case 'r':
      return generateRookMoves(row, col, piece.color);
    case 'q':
      return generateQueenMoves(row, col, piece.color);
    case 'k':
      return generateKingMoves(row, col, piece.color);
    default:
      return [];
  }
}

/**
 * Generate all pseudo‑legal moves for the active colour.
 *
 * @returns {Array<{from:{row,col}, to:{row,col}}>} All moves
 */
function generateAllMoves() {
  const moves = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const piece = boardState[r][c];
      if (piece && piece.color === activeColor) {
        const pieceMoves = generateMovesForPiece(r, c, piece);
        moves.push(...pieceMoves);
      }
    }
  }
  return moves;
}

/**
 * Pick a random move from the list of all available moves. This simulates
 * a dummy LLM suggestion.
 *
 * @returns {{from: string, to: string}|null} The suggested move or null if none
 */
function getLLMDummyMove() {
  const moves = generateAllMoves();
  if (moves.length === 0) return null;
  const randomIndex = Math.floor(Math.random() * moves.length);
  const move = moves[randomIndex];
  return {
    from: coordToSquare(move.from.row, move.from.col),
    to: coordToSquare(move.to.row, move.to.col)
  };
}

/**
 * Render the board based on the current boardState. Each square is a div
 * element with data attributes for row, col and algebraic square. Event
 * listeners are attached to handle piece selection and movement. Also
 * applies highlights for the LLM suggestion if available.
 */
function drawBoard() {
  // Clear any existing cells
  while (boardElement.firstChild) {
    boardElement.removeChild(boardElement.firstChild);
  }
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const cell = document.createElement('div');
      cell.classList.add('cell');
      // Alternate colours: light if (r + c) even, dark otherwise
      cell.classList.add((r + c) % 2 === 0 ? 'light' : 'dark');
      cell.dataset.row = r.toString();
      cell.dataset.col = c.toString();
      cell.dataset.square = coordToSquare(r, c);
      const piece = boardState[r][c];
      if (piece) {
        cell.textContent = PIECE_UNICODE[piece.char];
      }
      // Attach click handler
      cell.addEventListener('click', onCellClick);
      boardElement.appendChild(cell);
    }
  }
  // After the board is built we can apply highlights for the LLM suggestion
  highlightLLMMove();
}

/**
 * Highlight the squares associated with the LLM’s suggested move.
 */
function highlightLLMMove() {
  // Remove any existing LLM highlights
  document.querySelectorAll('.highlight-from, .highlight-to').forEach(el => {
    el.classList.remove('highlight-from');
    el.classList.remove('highlight-to');
  });
  if (!llmMove) return;
  const fromCell = document.querySelector(`[data-square="${llmMove.from}"]`);
  const toCell = document.querySelector(`[data-square="${llmMove.to}"]`);
  if (fromCell) fromCell.classList.add('highlight-from');
  if (toCell) toCell.classList.add('highlight-to');
}

/**
 * Clear selection and legal move highlights from all cells.
 */
function clearHighlights() {
  document.querySelectorAll('.selected').forEach(el => {
    el.classList.remove('selected');
  });
  document.querySelectorAll('.legal-move').forEach(el => {
    el.classList.remove('legal-move');
  });
}

/**
 * Highlight legal destination squares for the currently selected piece.
 */
function highlightLegalMoves() {
  legalMoves.forEach(move => {
    const cell = document.querySelector(`[data-row="${move.to.row}"][data-col="${move.to.col}"]`);
    if (cell) {
      cell.classList.add('legal-move');
    }
  });
}

/**
 * Handle a click on a board square. Implements piece selection and movement.
 */
function onCellClick(event) {
  if (userMoved) return; // No further moves once user has moved
  const cell = event.currentTarget;
  const row = parseInt(cell.dataset.row, 10);
  const col = parseInt(cell.dataset.col, 10);
  const piece = boardState[row][col];
  if (!selectedSquare) {
    // No piece selected yet: select if this cell contains a piece of the side to move
    if (piece && piece.color === activeColor) {
      selectedSquare = { row, col };
      legalMoves = generateMovesForPiece(row, col, piece);
      cell.classList.add('selected');
      highlightLegalMoves();
    }
    return;
  }
  // If clicking the same colour piece again, reselect that piece
  if (piece && piece.color === activeColor) {
    clearHighlights();
    selectedSquare = { row, col };
    legalMoves = generateMovesForPiece(row, col, piece);
    cell.classList.add('selected');
    highlightLegalMoves();
    return;
  }
  // Otherwise, see if this square is a legal destination
  const legal = legalMoves.find(m => m.to.row === row && m.to.col === col);
  if (legal) {
    // Execute the move: update board state
    const movingPiece = boardState[selectedSquare.row][selectedSquare.col];
    boardState[legal.to.row][legal.to.col] = movingPiece;
    boardState[selectedSquare.row][selectedSquare.col] = null;
    // Update the DOM and state
    drawBoard();
    userMoved = true;
    // Compare user move with LLM suggestion
    const userFrom = coordToSquare(selectedSquare.row, selectedSquare.col);
    const userTo = coordToSquare(legal.to.row, legal.to.col);
    compareWithLLM(userFrom, userTo);
  }
  // Clear selection and highlights regardless of whether the move was legal
  clearHighlights();
  selectedSquare = null;
  legalMoves = [];
}

/**
 * Compare the user’s move with the LLM suggestion and update the status text.
 *
 * @param {string} userFrom The origin square of the user’s move
 * @param {string} userTo The destination square of the user’s move
 */
function compareWithLLM(userFrom, userTo) {
  let match = false;
  if (llmMove) {
    match = (llmMove.from === userFrom && llmMove.to === userTo);
  }
  statusElement.textContent = match
    ? 'Great! Your move matches the LLM suggestion.'
    : 'Your move differs from the LLM suggestion.';
  if (llmMove) {
    llmElement.textContent = 'LLM suggested move: ' + llmMove.from + ' → ' + llmMove.to;
  } else {
    llmElement.textContent = 'LLM had no suggestion for this position.';
  }
}

/**
 * Convert a move in SAN (simple algebraic notation) from the model_move
 * column into a from/to coordinate pair. This implementation is simplified
 * and handles common move formats such as "e4", "Nf3", "Rxf7", "Qxg7+",
 * etc. It ignores check/mate symbols (+/#) and does not handle castling
 * or promotion. If the SAN string cannot be parsed or no matching move
 * exists in the generated move list, null is returned.
 *
 * @param {string} san The SAN string from the CSV (e.g. "Kg8", "e4", "Qd2").
 * @returns {{from: string, to: string}|null} An object with algebraic from/to squares
 */
function convertModelMove(san) {
  if (!san || san.trim() === '') return null;
  // Remove trailing check/mate markers and whitespace
  let moveStr = san.trim();
  moveStr = moveStr.replace(/[+#]+$/, '');
  // Handle castling (not supported by our pseudo‑legal generator)
  if (/^O-O(-O)?$/.test(moveStr) || /^0-0(-0)?$/.test(moveStr)) {
    return null;
  }
  // Determine if capture is indicated
  const isCapture = moveStr.includes('x');
  // Determine destination square
  const destMatch = moveStr.match(/([a-h][1-8])$/);
  if (!destMatch) return null;
  const destSquare = destMatch[1];
  // Determine piece type; default is pawn
  let pieceLetter = moveStr.charAt(0);
  let pieceType = 'p';
  const pieces = { K: 'k', Q: 'q', R: 'r', B: 'b', N: 'n' };
  if (pieces[pieceLetter]) {
    pieceType = pieces[pieceLetter];
  }
  // Generate all pseudo‑legal moves and find ones matching destination and piece
  const moves = generateAllMoves();
  for (const m of moves) {
    const fromSq = coordToSquare(m.from.row, m.from.col);
    const toSq = coordToSquare(m.to.row, m.to.col);
    // Check destination
    if (toSq !== destSquare) continue;
    const movingPiece = boardState[m.from.row][m.from.col];
    if (!movingPiece) continue;
    // Check piece type
    if (movingPiece.type !== pieceType) continue;
    // If capture indicated, ensure destination was occupied by opponent at start
    if (isCapture) {
      const target = boardState[m.to.row][m.to.col];
      if (!target || target.color === movingPiece.color) continue;
    }
    // If not capture, ensure destination is empty
    if (!isCapture) {
      const target = boardState[m.to.row][m.to.col];
      if (target) continue;
    }
    return { from: fromSq, to: toSq };
  }
  // If no match found, return null
  return null;
}

/**
 * Load a dataset entry at the given index. Reads the FEN and model_move
 * from the global ENTRIES array (if available), updates the input box,
 * parses the FEN, draws the board and computes the LLM suggestion. If the
 * model_move is empty or cannot be parsed, falls back to a random dummy
 * suggestion. Updates instruction text accordingly. Assumes the index is
 * valid (0 ≤ index < ENTRIES.length).
 *
 * @param {number} index Index into the ENTRIES array
 */
function loadEntry(index) {
  currentEntryIndex = index;
  const entry = ENTRIES[index];
  const fen = entry.fen;
  fenInput.value = fen;
  // Reset state
  userMoved = false;
  llmMove = null;
  selectedSquare = null;
  legalMoves = [];
  statusElement.textContent = '';
  llmElement.textContent = '';
  instructionElement.textContent = '';
  // Parse FEN and draw board
  if (!parseFEN(fen)) {
    statusElement.textContent = 'Invalid FEN in dataset entry.';
    boardElement.innerHTML = '';
    return;
  }
  drawBoard();
  // Determine LLM move from the model_move SAN
  const san = entry.model_move ? entry.model_move.trim() : '';
  let parsedMove = convertModelMove(san);
  if (!parsedMove) {
    // If model_move is empty or parsing failed, use dummy random move
    parsedMove = getLLMDummyMove();
  }
  llmMove = parsedMove;
  highlightLLMMove();
  // Instruction about turn
  const side = activeColor === 'w' ? 'White' : 'Black';
  instructionElement.textContent =
    'Position ' + (index + 1) + ' of ' + ENTRIES.length + '. It is ' + side + "'s move. Select a piece to play your move.";
}

/**
 * Load the next entry from the dataset. Advances the currentEntryIndex
 * cyclically and calls loadEntry. If there is no dataset, this function
 * does nothing.
 */
function loadNextEntry() {
  if (typeof ENTRIES === 'undefined' || !Array.isArray(ENTRIES) || ENTRIES.length === 0) {
    return;
  }
  const nextIndex = (currentEntryIndex + 1) % ENTRIES.length;
  loadEntry(nextIndex);
}

/**
 * Parse a CSV string into an array of entry objects with `fen` and
 * `model_move` properties.  The parser handles quoted fields, escaped
 * quotes and newlines within fields according to RFC 4180.  Only the
 * second and third columns of each row (index 1 and 2) are used.  The
 * header row is skipped.
 *
 * @param {string} csv The raw CSV text
 * @returns {Array<{fen: string, model_move: string}>} Parsed entries
 */
function parseCSV(csv) {
  const entries = [];
  let i = 0;
  const len = csv.length;
  let field = '';
  let row = [];
  let inQuotes = false;
  while (i < len) {
    const char = csv[i];
    if (inQuotes) {
      if (char === '"') {
        // Peek at next char to see if this is an escaped quote
        if (i + 1 < len && csv[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += char;
      }
    } else {
      if (char === '"') {
        inQuotes = true;
      } else if (char === ',') {
        row.push(field);
        field = '';
      } else if (char === '\r') {
        // ignore carriage returns
      } else if (char === '\n') {
        row.push(field);
        field = '';
        // Process the row if it has at least two fields (skip header)
        if (row.length >= 3) {
          const fen = row[1] ? row[1].trim() : '';
          const move = row[2] ? row[2].trim() : '';
          entries.push({ fen: fen, model_move: move });
        }
        row = [];
      } else {
        field += char;
      }
    }
    i++;
  }
  // Handle the last line if it doesn't end with newline
  if (field !== '' || row.length) {
    row.push(field);
    if (row.length >= 3 && entries.length >= 0) {
      const fen = row[1] ? row[1].trim() : '';
      const move = row[2] ? row[2].trim() : '';
      entries.push({ fen: fen, model_move: move });
    }
  }
  // Remove header entry if present (e.g., first row may be header)
  if (entries.length > 0 && entries[0].fen.toLowerCase() === 'fen') {
    entries.shift();
  }
  return entries;
}


/**
 * Event handler for the “Load Position” button. Parses the FEN string,
 * resets the internal state, draws the board, generates the LLM suggestion
 * and updates the instructional text.
 */
function onLoadButtonClick() {
  const fen = fenInput.value;
  // Reset messages and state
  userMoved = false;
  llmMove = null;
  selectedSquare = null;
  legalMoves = [];
  statusElement.textContent = '';
  llmElement.textContent = '';
  instructionElement.textContent = '';
  // Parse FEN
  if (!parseFEN(fen)) {
    statusElement.textContent = 'Invalid FEN. Please enter a valid FEN string.';
    boardElement.innerHTML = '';
    return;
  }
  // Draw the board for the new position
  drawBoard();
  // Generate an LLM suggestion
  llmMove = getLLMDummyMove();
  if (llmMove) {
    // We highlight the LLM move here; the actual message is shown after the user moves
    highlightLLMMove();
  }
  // Show whose turn it is
  const side = activeColor === 'w' ? 'White' : 'Black';
  instructionElement.textContent = 'It is ' + side + '\'s move. Select a piece to play your move.';
}

// Attach the load button’s click handler
loadButton.addEventListener('click', onLoadButtonClick);

// Initialise the application on page load. If a dataset is available via
// ENTRIES, load the first entry and show the navigation controls. Otherwise
// fall back to the default FEN in the input field.
window.addEventListener('DOMContentLoaded', () => {
  // Async wrapper so we can await fetch
  (async () => {
    // Attempt to fetch a data.csv file from the same directory. If the
    // fetch succeeds and the file contains valid entries, override
    // ENTRIES with the parsed dataset. Some browsers may restrict fetch
    // on the file:// protocol; in that case this silently fails.
    try {
      const response = await fetch('./data.csv');
      console.log(response,'response')
      if (response && response.ok) {
        const text = await response.text();
        const parsed = parseCSV(text);
        if (parsed && parsed.length > 0) {
          window.ENTRIES = parsed;
        }
      }
    } catch (err) {
      console.log(err,'error')
      // Swallow any fetch or parse errors. We fall back to whatever
      // ENTRIES may already contain (e.g., from data.js) or manual FEN input.
    }
    // Attach click handler for the Next button regardless of whether a dataset is present.
    if (nextButton) {
      nextButton.addEventListener('click', () => {
        loadNextEntry();
      });
    }
    // Always show the navigation controls so the Next button is visible
    if (navigationElement) {
      navigationElement.style.display = 'block';
    }
    // Now proceed to initialise the UI: if we have a dataset, load it; otherwise fall back
    if (typeof ENTRIES !== 'undefined' && Array.isArray(ENTRIES) && ENTRIES.length > 0) {
      // Load the first entry
      loadEntry(0);
    } else {
      // No dataset: initialise board from default FEN in input box
      const initialFen = fenInput.value;
      if (parseFEN(initialFen)) {
        drawBoard();
        llmMove = getLLMDummyMove();
        highlightLLMMove();
        const side = activeColor === 'w' ? 'White' : 'Black';
        instructionElement.textContent = 'It is ' + side + '\'s move. Select a piece to play your move.';
      }
    }
  })();
});