# Chess FEN Trainer

This project is a simple, self‑contained web application that allows you to
load any chess position using a **Forsyth–Edwards Notation (FEN)** string
(sometimes written as “Forsyth‑Edwards Notation”)【115677098837149†L133-L165】, display that position on a board and make the
next move. A dummy Large‑Language‑Model (LLM) picks a random pseudo‑legal
move for the side to move and highlights it on the board. After you play
your move, the application compares it to the LLM suggestion and lets you
know if they match.

## Features

* **FEN input** – enter any valid FEN string and click **Load Position** to
  draw the position on the board. Only the piece placement and side‑to‑move
  fields are used; castling rights, en‑passant targets and move counters are
  ignored.
* **Interactive board** – click a piece to see its pseudo‑legal moves
  highlighted in green. Click a highlighted square to make the move.
* **Dummy LLM suggestion** – after loading a position the app chooses a
  random pseudo‑legal move for the side to move and highlights it in gold.
* **Move comparison** – after you play your move the app displays whether it
  matches the LLM’s suggestion and shows what the LLM would have played.

## Limitations

* Move generation is **pseudo‑legal**: castling, en‑passant and check
  detection are not implemented. Moves that leave or put your king in check
  might still be considered legal by the app. For educational purposes this
  is usually sufficient.
* Only the first two fields of the FEN string are parsed (piece placement
  and active colour). The remaining fields are ignored.
* The LLM suggestion is intentionally simplistic and random; it doesn’t
  evaluate the position.

## Running the application

1. Extract or clone the `chess_app` folder somewhere on your computer.
2. Open the file **`index.html`** in a modern web browser (e.g. Chrome,
   Firefox, Safari). The application is fully self‑contained and does not
   require an internet connection or a local web server.
3. Enter a FEN string in the input box or leave the provided example, then
   click **Load Position**. The board will update and the dummy LLM move will
   be highlighted.
4. Click on a piece belonging to the side to move to see its available moves
   and click again on the destination square to make your move. A message
   beneath the board will indicate whether your move matches the LLM’s
   suggestion and display what the suggestion was.

That’s it! Feel free to experiment by loading different positions and
comparing your moves against the random suggestions.