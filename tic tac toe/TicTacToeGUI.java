import java.awt.*;
import java.awt.event.*;
import java.util.Random;
import javax.swing.*;

public class TicTacToeGUI extends JFrame {
    private BoardPanel boardPanel;
    private char[][] board = new char[3][3];
    private boolean playerTurn = true; // true = X, false = O/Bot
    private boolean vsBot = false;
    private int botDifficulty = 1; // 1=Easy, 2=Medium, 3=Hard
    private char playerMark = 'X', botMark = 'O';
    private int[][] winLine = null; // { {row1,col1}, {row2,col2} }
    private boolean gameOver = false;

    public TicTacToeGUI() {
        setTitle("Tic Tac Toe");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(420, 520);
        setLayout(new BorderLayout());

        JLabel statusLabel = new JLabel("Welcome to Tic Tac Toe!", SwingConstants.CENTER);
        statusLabel.setFont(new Font("Arial", Font.BOLD, 20));
        add(statusLabel, BorderLayout.NORTH);

        boardPanel = new BoardPanel();
        add(boardPanel, BorderLayout.CENTER);

        JPanel controlPanel = new JPanel();
        JButton restartBtn = new JButton("Restart");
        restartBtn.addActionListener(e -> setupGame());
        controlPanel.add(restartBtn);

        JButton modeBtn = new JButton("Change Mode");
        modeBtn.addActionListener(e -> chooseMode());
        controlPanel.add(modeBtn);

        add(controlPanel, BorderLayout.SOUTH);

        boardPanel.setStatusLabel(statusLabel);

        // Prompt for mode selection at startup
        chooseMode();

        setVisible(true);
    }

    private void setupGame() {
        for (char[] row : board) java.util.Arrays.fill(row, '-');
        playerTurn = true;
        winLine = null;
        gameOver = false;
        boardPanel.setEnabled(true);
        boardPanel.repaint();
        boardPanel.setStatusText(vsBot ? "Your turn (X)" : "Player X's turn");
        if (vsBot && botMark == 'X') {
            playerTurn = false;
            boardPanel.setStatusText("Bot's turn (" + botMark + ")");
            javax.swing.Timer t = new javax.swing.Timer(500, e -> botMove());
            t.setRepeats(false);
            t.start();
        }
    }

    private void chooseMode() {
        Object[] options = {"Pass n Play (2 players)", "Play vs Bot"};
        int mode = JOptionPane.showOptionDialog(this, "Choose mode:", "Game Mode",
                JOptionPane.DEFAULT_OPTION, JOptionPane.QUESTION_MESSAGE, null, options, options[0]);
        if (mode == 1) {
            vsBot = true;
            Object[] diffs = {"Easy", "Medium", "Hard"};
            int diff = JOptionPane.showOptionDialog(this, "Bot Difficulty:", "Choose Difficulty",
                    JOptionPane.DEFAULT_OPTION, JOptionPane.QUESTION_MESSAGE, null, diffs, diffs[0]);
            botDifficulty = diff + 1;
            playerMark = 'X';
            botMark = 'O';
        } else {
            vsBot = false;
        }
        setupGame();
    }

    private void handleMove(int row, int col) {
        if (board[row][col] != '-' || isGameOver()) return;
        if (vsBot) {
            if (!playerTurn) return;
            makeMove(row, col, playerMark);
            if (!isGameOver()) {
                playerTurn = false;
                boardPanel.setStatusText("Bot's turn (" + botMark + ")");
                javax.swing.Timer t = new javax.swing.Timer(500, e -> botMove());
                t.setRepeats(false);
                t.start();
            }
        } else {
            char mark = playerTurn ? 'X' : 'O';
            makeMove(row, col, mark);
            if (!isGameOver()) {
                playerTurn = !playerTurn;
                boardPanel.setStatusText("Player " + (playerTurn ? "X" : "O") + "'s turn");
            }
        }
    }

    private void makeMove(int row, int col, char mark) {
        board[row][col] = mark;
        boardPanel.repaint();
        if (hasWon(mark)) {
            winLine = getWinLine(mark); // Set winLine only after a real win
            String winnerMsg = (vsBot && mark == botMark) ? "Bot (" + mark + ")" :
                               (vsBot && mark == playerMark) ? "You (" + mark + ")" :
                               "Player " + mark;
            boardPanel.setStatusText(winnerMsg + " wins!");
            boardPanel.setEnabled(false);
            gameOver = true;
            showEndDialog(winnerMsg + " has won!");
        } else if (isDraw()) {
            boardPanel.setStatusText("It's a draw!");
            boardPanel.setEnabled(false);
            gameOver = true;
            showEndDialog("It's a draw!");
        }
    }

    private void botMove() {
        int[] move;
        if (botDifficulty == 1) move = randomMove();
        else if (botDifficulty == 2) move = mediumBotMove(botMark, playerMark);
        else move = minimaxMove(botMark, playerMark);
        makeMove(move[0], move[1], botMark);
        if (!isGameOver()) {
            playerTurn = true;
            boardPanel.setStatusText("Your turn (" + playerMark + ")");
        }
    }

    private boolean isGameOver() {
        return (winLine != null) || isDraw() || gameOver;
    }

    // Only checks for win, does not set winLine
    private boolean hasWon(char c) {
        // Rows
        for (int i = 0; i < 3; i++)
            if (board[i][0] == c && board[i][1] == c && board[i][2] == c)
                return true;
        // Columns
        for (int i = 0; i < 3; i++)
            if (board[0][i] == c && board[1][i] == c && board[2][i] == c)
                return true;
        // Diagonals
        if (board[0][0] == c && board[1][1] == c && board[2][2] == c)
            return true;
        if (board[0][2] == c && board[1][1] == c && board[2][0] == c)
            return true;
        return false;
    }

    // Returns the win line if player c has won, else null
    private int[][] getWinLine(char c) {
        for (int i = 0; i < 3; i++)
            if (board[i][0] == c && board[i][1] == c && board[i][2] == c)
                return new int[][]{{i,0},{i,2}};
        for (int i = 0; i < 3; i++)
            if (board[0][i] == c && board[1][i] == c && board[2][i] == c)
                return new int[][]{{0,i},{2,i}};
        if (board[0][0] == c && board[1][1] == c && board[2][2] == c)
            return new int[][]{{0,0},{2,2}};
        if (board[0][2] == c && board[1][1] == c && board[2][0] == c)
            return new int[][]{{0,2},{2,0}};
        return null;
    }

    private boolean isDraw() {
        for (char[] row : board)
            for (char cell : row)
                if (cell == '-') return false;
        return winLine == null;
    }

    // Bot: Easy (random)
    private int[] randomMove() {
        java.util.List<int[]> moves = new java.util.ArrayList<>();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (board[i][j] == '-') moves.add(new int[]{i, j});
        return moves.get(new Random().nextInt(moves.size()));
    }

    // Bot: Medium (block or win, else random)
    private int[] mediumBotMove(char bot, char human) {
        // Try to win
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (board[i][j] == '-') {
                    board[i][j] = bot;
                    boolean win = hasWon(bot);
                    board[i][j] = '-';
                    if (win) return new int[]{i, j};
                }
        // Try to block
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (board[i][j] == '-') {
                    board[i][j] = human;
                    boolean block = hasWon(human);
                    board[i][j] = '-';
                    if (block) return new int[]{i, j};
                }
        // Else random
        return randomMove();
    }

    // Bot: Hard (Minimax)
    private int[] minimaxMove(char bot, char human) {
        int bestScore = Integer.MIN_VALUE;
        int[] move = {-1, -1};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (board[i][j] == '-') {
                    board[i][j] = bot;
                    int score = minimax(false, bot, human);
                    board[i][j] = '-';
                    if (score > bestScore) {
                        bestScore = score;
                        move = new int[]{i, j};
                    }
                }
        return move;
    }

    private int minimax(boolean isMax, char bot, char human) {
        if (hasWon(bot)) return 1;
        if (hasWon(human)) return -1;
        if (isDraw()) return 0;
        int best = isMax ? Integer.MIN_VALUE : Integer.MAX_VALUE;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (board[i][j] == '-') {
                    board[i][j] = isMax ? bot : human;
                    int score = minimax(!isMax, bot, human);
                    board[i][j] = '-';
                    if (isMax) best = Math.max(best, score);
                    else best = Math.min(best, score);
                }
        return best;
    }

    private void showEndDialog(String message) {
        // Use SwingUtilities.invokeLater to ensure strike line is painted first
        SwingUtilities.invokeLater(() -> {
            boardPanel.repaint();
            Object[] options = {"Play Again", "Change Mode"};
            int result = JOptionPane.showOptionDialog(this,
                    "<html><center><b>" + message + "</b><br><br>What would you like to do?</center></html>",
                    "Game Over", JOptionPane.DEFAULT_OPTION, JOptionPane.INFORMATION_MESSAGE,
                    null, options, options[0]);
            if (result == 0) {
                setupGame();
            } else if (result == 1) {
                chooseMode();
            }
        });
    }

    // Custom JPanel for drawing the board, X/O, and win line
    class BoardPanel extends JPanel {
        private JLabel statusLabel;
        public BoardPanel() {
            setPreferredSize(new Dimension(400, 400));
            addMouseListener(new MouseAdapter() {
                public void mousePressed(MouseEvent e) {
                    if (!isEnabled() || isGameOver()) return;
                    int size = Math.min(getWidth(), getHeight());
                    int cellSize = size / 3;
                    int col = e.getX() / cellSize;
                    int row = e.getY() / cellSize;
                    if (row >= 0 && row < 3 && col >= 0 && col < 3)
                        handleMove(row, col);
                }
            });
        }

        public void setStatusLabel(JLabel label) {
            this.statusLabel = label;
        }

        public void setStatusText(String text) {
            if (statusLabel != null) statusLabel.setText(text);
        }

        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            int size = Math.min(getWidth(), getHeight());
            int cellSize = size / 3;
            Graphics2D g2 = (Graphics2D) g;
            g2.setStroke(new BasicStroke(4));
            // Draw grid
            g2.setColor(Color.BLACK);
            for (int i = 1; i < 3; i++) {
                g2.drawLine(i * cellSize, 0, i * cellSize, size);
                g2.drawLine(0, i * cellSize, size, i * cellSize);
            }
            // Draw X and O
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                    int x = j * cellSize;
                    int y = i * cellSize;
                    if (board[i][j] == 'X') {
                        g2.setColor(new Color(44, 62, 80));
                        g2.drawLine(x + 20, y + 20, x + cellSize - 20, y + cellSize - 20);
                        g2.drawLine(x + cellSize - 20, y + 20, x + 20, y + cellSize - 20);
                    } else if (board[i][j] == 'O') {
                        g2.setColor(new Color(211, 84, 0));
                        g2.drawOval(x + 20, y + 20, cellSize - 40, cellSize - 40);
                    }
                }
            // Draw strike-through win line
            if (winLine != null) {
                g2.setColor(Color.RED);
                g2.setStroke(new BasicStroke(8, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
                int r1 = winLine[0][0], c1 = winLine[0][1];
                int r2 = winLine[1][0], c2 = winLine[1][1];
                int x1 = c1 * cellSize + cellSize / 2;
                int y1 = r1 * cellSize + cellSize / 2;
                int x2 = c2 * cellSize + cellSize / 2;
                int y2 = r2 * cellSize + cellSize / 2;
                g2.drawLine(x1, y1, x2, y2);
            }
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(TicTacToeGUI::new);
    }
}
