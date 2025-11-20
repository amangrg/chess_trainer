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
// Whether the LLM's suggested move is currently visible on the board.
let llmVisible = false;

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
const fullReplyElement = document.getElementById('full-reply');
const llmEvalElement = document.getElementById('llm-eval');
const userMoveElement = document.getElementById('user-move');


// Navigation DOM elements (may not exist in older versions)
const navigationElement = document.getElementById('navigation');
const nextButton = document.getElementById('next-button');
// const fullReplyElement is declared above

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
  if (llmVisible) highlightLLMMove();
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
    // Now that the user has played, reveal the LLM suggestion on the board
    // and mark it so the player can compare visually.
    llmVisible = true;
    highlightLLMMove();
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
  // Show the user's move in the labelled output area
  if (userMoveElement) userMoveElement.textContent = userFrom + ' → ' + userTo;
  if (llmMove) {
    llmElement.textContent = 'LLM suggested move: ' + llmMove.from + ' → ' + llmMove.to;
  } else {
    llmElement.textContent = 'LLM had no suggestion for this position.';
  }
  // Show full_reply for gpt20b and show LLM move evaluation for all models
  try {
    const modelSelect = document.getElementById('model-select');
    const entry = ENTRIES && ENTRIES[currentEntryIndex];
    // Show full_reply only for gpt20b
    if (modelSelect && modelSelect.value === 'gpt20b.csv' && entry && fullReplyElement) {
      fullReplyElement.textContent = entry.full_reply || '';
    } else if (fullReplyElement) {
      fullReplyElement.textContent = '';
    }
    // Show LLM move evaluation for all models
    if (entry && llmEvalElement) {
      if (entry.good === true) {
        llmEvalElement.textContent = '✅ LLM predicted move was the best move.';
        llmEvalElement.style.color = '#059669';
      } else if (entry.good === false) {
        llmEvalElement.textContent = '❌ LLM predicted move was NOT the best move.';
        llmEvalElement.style.color = '#e11d48';
      } else {
        llmEvalElement.textContent = '';
        llmEvalElement.style.color = '';
      }
    }
  } catch (e) {
    console.warn('Error displaying full_reply or llmEval:', e);
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
  if (fullReplyElement) fullReplyElement.textContent = '';
  if (llmEvalElement) llmEvalElement.textContent = '';
  if (userMoveElement) userMoveElement.textContent = '';
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
  // Do not show the LLM suggestion while the user is choosing their move.
  llmVisible = false;
  // Instruction about turn
  const side = activeColor === 'w' ? 'White' : 'Black';
  instructionElement.textContent =
    'It is ' + side + "'s move. Select a piece to play your move.";
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
    // Parse a CSV string into an array of entry objects with `fen`,
    // `model_move` and `legal` properties. The parser handles quoted
    // fields, escaped quotes and newlines within fields according to
    // RFC 4180. We will skip any row where the `legal` column is
    // explicitly "FALSE" (case-insensitive).
    /**
     * @param {string} csv The raw CSV text
     * @returns {Array<{fen: string, model_move: string, legal: boolean}>} Parsed entries
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
            // Process the row if it has at least the expected fields
            // We expect: id, fen, model_move, full_reply, legal, ...
            if (row.length >= 5) {
              const fen = row[1] ? row[1].trim() : '';
              const move = row[2] ? row[2].trim() : '';
              const legalRaw = row[4] ? row[4].trim() : '';
              const legal = !(legalRaw.toLowerCase && legalRaw.toLowerCase() === 'false');
              if (legal) {
                const reply = row[3] ? row[3].trim() : '';
                const goodRaw = row[5] ? row[5].trim() : '';
                const good = (goodRaw.toLowerCase && goodRaw.toLowerCase() === 'true');
                entries.push({ fen: fen, model_move: move, legal: true, full_reply: reply, good: good });
              }
            } else if (row.length >= 3) {
              // Backwards-compatible: if fewer columns are present, accept the row
              const fen = row[1] ? row[1].trim() : '';
              const move = row[2] ? row[2].trim() : '';
              const goodRaw = row[5] ? row[5].trim() : '';
              const good = (goodRaw.toLowerCase && goodRaw.toLowerCase() === 'true');
              entries.push({ fen: fen, model_move: move, legal: true, full_reply: '', good: good });
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
        if (row.length >= 5) {
          const fen = row[1] ? row[1].trim() : '';
          const move = row[2] ? row[2].trim() : '';
          const legalRaw = row[4] ? row[4].trim() : '';
          const legal = !(legalRaw.toLowerCase && legalRaw.toLowerCase() === 'false');
          if (legal) {
            const reply = row[3] ? row[3].trim() : '';
            const goodRaw = row[5] ? row[5].trim() : '';
            const good = (goodRaw.toLowerCase && goodRaw.toLowerCase() === 'true');
            entries.push({ fen: fen, model_move: move, legal: true, full_reply: reply, good: good });
          }
        } else if (row.length >= 3) {
          const fen = row[1] ? row[1].trim() : '';
          const move = row[2] ? row[2].trim() : '';
          const goodRaw = row[5] ? row[5].trim() : '';
          const good = (goodRaw.toLowerCase && goodRaw.toLowerCase() === 'true');
          entries.push({ fen: fen, model_move: move, legal: true, full_reply: '', good: good });
        }
      }
      // Remove header entry if present (e.g., first row may be header)
      if (entries.length > 0 && entries[0].fen && entries[0].fen.toLowerCase() === 'fen') {
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
  if (fullReplyElement) fullReplyElement.textContent = '';
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
  // Don't highlight the suggestion while the user is selecting their move.
  llmVisible = false;
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
    // Load dataset according to the selected model CSV (or fallback to an
    // already-included `data.js` ENTRIES variable). This supports selecting
    // different CSV files (gpt20b.csv, mistral7b.csv, etc.) from the
    // dropdown added to the page.
    async function loadCSVFile(filename) {
      try {
        const response = await fetch(filename);
        if (!response || !response.ok) {
          console.warn('Could not fetch', filename, response);
          return null;
        }
        const text = await response.text();
        const parsed = parseCSV(text);
        if (parsed && parsed.length > 0) return parsed;
        return null;
      } catch (err) {
        console.warn('Error fetching/parsing', filename, err);
        return null;
      }
    }

    // Helper to read the current UI selection and load data accordingly.
    async function loadSelectedModelData() {
      const select = document.getElementById('model-select');
      const customInput = document.getElementById('model-custom');
      console.log(select.value,customInput) 
      if (!select) return;
      const val = select.value;
      if (val === 'other') {
        const custom = customInput && customInput.value ? customInput.value.trim() : '';
        
        if (custom) {
          const parsed = await loadCSVFile(custom);
          if (parsed) {
            window.ENTRIES = parsed;
            loadEntry(0);
            return;
          }
          // If custom file was provided but failed to load, clear ENTRIES so
          // we do not fall back to a previously-loaded dataset.
          window.ENTRIES = [];
          instructionElement.textContent = 'Could not load "' + custom + '" — check filename and server. Showing manual FEN.';
          console.warn('Failed to load custom CSV:', custom);
        }
        // If no custom file or load failed, fall through to check existing ENTRIES
      } else if (val) {
        console.log(val,'file name')
        const parsed = await loadCSVFile(val);
        if (parsed) {
          window.ENTRIES = parsed;
          loadEntry(0);
          return;
        }
        // Clear any existing ENTRIES if the requested file failed to load
        window.ENTRIES = [];
        instructionElement.textContent = 'Could not load "' + val + '" — check that the file exists on the server.';
        console.warn('Failed to load CSV for selection:', val);
      }
      // If we get here, either fetch failed or there was no selection that
      // produced entries. If `ENTRIES` was already defined (e.g. via data.js)
      // use it; otherwise fall back to the manual FEN input behaviour below.
      if (typeof ENTRIES !== 'undefined' && Array.isArray(ENTRIES) && ENTRIES.length > 0) {
        loadEntry(0);
      } else {
        // No dataset available: initialise board from default FEN in input box
        const initialFen = fenInput.value;
        if (parseFEN(initialFen)) {
          drawBoard();
          llmMove = getLLMDummyMove();
          // Do not highlight while the user has not yet played
          llmVisible = false;
          const side = activeColor === 'w' ? 'White' : 'Black';
          instructionElement.textContent = 'It is ' + side + '\'s move. Select a piece to play your move.';
        }
      }
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
    // Wire up the model select UI (show/hide custom input and react to changes)
    const modelSelect = document.getElementById('model-select');
    const modelCustom = document.getElementById('model-custom');
    if (modelSelect) {
      modelSelect.addEventListener('change', (e) => {
        // Immediately reset UI to first-entry/loading state when model changes
        currentEntryIndex = 0;
        userMoved = false;
        selectedSquare = null;
        legalMoves = [];
        boardElement.innerHTML = '';
        statusElement.textContent = '';
        llmElement.textContent = '';
        instructionElement.textContent = 'Loading selected model...';
  if (fullReplyElement) fullReplyElement.textContent = '';
  if (llmEvalElement) llmEvalElement.textContent = '';
  if (userMoveElement) userMoveElement.textContent = '';
  llmVisible = false;

        if (modelCustom) {
          if (modelSelect.value === 'other') {
            modelCustom.style.display = 'inline-block';
            modelCustom.focus();
          } else {
            modelCustom.style.display = 'none';
          }
        }
        // Load dataset for newly selected model (this will call loadEntry(0) when ready)
        loadSelectedModelData();
      });
    }

    // If user types a custom filename and presses Enter, attempt to load it
    if (modelCustom) {
      modelCustom.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') {
          loadSelectedModelData();
        }
      });
    }
    console.log("Tejas")

    // Initial load according to current selection (defaults to gpt20b.csv in the HTML)
    await loadSelectedModelData();
  })();
});