# ❌ Tic Tac Toe ⭕

An interactive, feature-rich 2D Tic Tac Toe game built using Java's Swing GUI library. The game supports local two-player ("Pass n' Play") mode as well as a single-player mode against an intelligent computer AI.

---

## 🤖 AI Bots & Minimax Algorithm

When playing vs. the Computer Bot, you can choose from three difficulties:
1. **Easy**: The bot places its mark in completely random empty cells on the grid.
2. **Medium**: The bot is slightly smarter. It will inspect the board and instantly claim a winning cell if available. If not, it checks if the human player has 2 in a row and blocks their win; otherwise, it moves randomly.
3. **Hard (Unbeatable)**: The bot is powered by the **Minimax Algorithm**, an advanced game theory decision-making tree that recursively searches all possible moves. The bot evaluates every possible outcome to maximize its score and minimize the player's score, making it mathematically **unbeatable** (the best you can achieve is a draw!).

---

## 📂 Project Structure

- 📄 **`TicTacToeGUI.java`**: The complete application code, including:
  - Game state machines.
  - Interactive grid cells.
  - Graphical render loop (uses Swing graphics to paint the clean grid lines, smooth X drawing vectors, and O ovals).
  - The AI Bot algorithms (Easy, Medium, and Minimax).
  - Strike-through red winning line overlay calculation.

---

## 🚀 How to Compile and Run

To compile and launch the game, ensure you have the Java Development Kit (JDK 17+) installed:

1. Open your terminal in the `tic tac toe` folder.
2. Compile the Java source file:
   ```bash
   javac TicTacToeGUI.java
   ```
3. Run the compiled application:
   ```bash
   java TicTacToeGUI
   ```

---

## 🎮 Game Controls & Customization
- **Change Mode**: Click the **"Change Mode"** button at the bottom of the window to swap between multiplayer and bot difficulties without closing the app.
- **Restart**: Tap **"Restart"** to instantly clear the board grid and launch a new match.
