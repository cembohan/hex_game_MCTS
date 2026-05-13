// ── State ────────────────────────────────────────────────────────────────
let boardSize = 11;          // overridden by /config
const hexWidth = 40;
const hexHeight = 46.188;

const boardEl = document.getElementById('board');
const startBtn = document.getElementById('start-btn');
const tempSlider = document.getElementById('temp-slider');
const tempVal = document.getElementById('temp-val');
const colorSelect = document.getElementById('color-select');
const statusPanel = document.getElementById('status-panel');
const turnIndicator = document.getElementById('turn-indicator');
const swapBtn = document.getElementById('swap-btn');
const configInfo = document.getElementById('config-info');

let pollInterval = null;
let currentTurn = 0;
let isHumanTurn = false;
let gameIsOver = false;
let hasHumanPlayer = true;   // false when AI-vs-AI

// ── Temperature slider ──────────────────────────────────────────────────
tempSlider.addEventListener('input', (e) => {
    tempVal.textContent = e.target.value;
});

// ── Fetch server config on load ─────────────────────────────────────────
async function loadConfig() {
    try {
        const resp = await fetch('/config');
        const cfg = await resp.json();
        boardSize = cfg.board_size || 11;

        const p1Human = cfg.p1_type === 'human';
        const p2Human = cfg.p2_type === 'human';
        hasHumanPlayer = p1Human || p2Human;

        // Build info string
        const p1Label = p1Human ? '🧑 Human' : `🤖 ${cfg.p1_name}`;
        const p2Label = p2Human ? '🧑 Human' : `🤖 ${cfg.p2_name}`;
        if (configInfo) {
            configInfo.textContent = `${cfg.board_size}×${cfg.board_size}  |  Red: ${p1Label}  vs  Blue: ${p2Label}`;
        }

        // If both are AI, hide the colour picker
        if (!hasHumanPlayer) {
            const colorGroup = colorSelect ? colorSelect.closest('.control-group') : null;
            if (colorGroup) colorGroup.style.display = 'none';
        } else if (p1Human && !p2Human) {
            if (colorSelect) colorSelect.value = 'RED';
        } else if (!p1Human && p2Human) {
            if (colorSelect) colorSelect.value = 'BLUE';
        }
    } catch (e) {
        console.warn('Could not fetch /config, using defaults', e);
    }
    initBoard();
}

// ── Board rendering ─────────────────────────────────────────────────────
function initBoard() {
    boardEl.innerHTML = '';

    const totalWidth = hexWidth * boardSize + (boardSize * hexWidth / 2);
    const totalHeight = hexHeight * 0.75 * boardSize + hexHeight * 0.25;

    boardEl.style.width = `${totalWidth}px`;
    boardEl.style.height = `${totalHeight}px`;

    // Side indicators
    const leftSide = document.createElement('div');
    leftSide.className = 'board-side-left';
    boardEl.appendChild(leftSide);

    const rightSide = document.createElement('div');
    rightSide.className = 'board-side-right';
    boardEl.appendChild(rightSide);

    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const hex = document.createElement('div');
            hex.className = 'hex';
            hex.dataset.r = r;
            hex.dataset.c = c;

            const x = c * hexWidth + (r * hexWidth / 2);
            const y = r * hexHeight * 0.75;

            hex.style.left = `${x}px`;
            hex.style.top = `${y}px`;

            hex.addEventListener('click', () => onHexClick(r, c));
            boardEl.appendChild(hex);
        }
    }
}

// ── Hex click (human move) ──────────────────────────────────────────────
async function onHexClick(r, c) {
    if (!isHumanTurn || gameIsOver) return;

    try {
        const response = await fetch('/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x: r, y: c })
        });

        if (response.ok) {
            const hex = document.querySelector(`.hex[data-r="${r}"][data-c="${c}"]`);
            if (hex && colorSelect) {
                hex.className = `hex ${colorSelect.value.toLowerCase()}`;
            }
            isHumanTurn = false;
            updateStatusUI();
            swapBtn.classList.add('hidden');
        }
    } catch (err) {
        console.error(err);
    }
}

// ── Swap rule ───────────────────────────────────────────────────────────
swapBtn.addEventListener('click', async () => {
    if (!isHumanTurn || gameIsOver) return;

    try {
        const response = await fetch('/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ swap: true })
        });

        if (response.ok) {
            isHumanTurn = false;
            updateStatusUI();
            swapBtn.classList.add('hidden');
        }
    } catch (err) {
        console.error(err);
    }
});

// ── Start game ──────────────────────────────────────────────────────────
startBtn.addEventListener('click', async () => {
    initBoard();
    gameIsOver = false;

    const humanColor = colorSelect ? colorSelect.value : 'RED';
    const temp = parseFloat(tempSlider.value);

    try {
        await fetch('/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ human_plays: humanColor, temperature: temp })
        });

        statusPanel.classList.remove('hidden');
        turnIndicator.textContent = "Game started. Waiting...";

        if (pollInterval) clearInterval(pollInterval);
        // AI-vs-AI needs faster polling since no human latency
        const interval = hasHumanPlayer ? 500 : 300;
        pollInterval = setInterval(pollState, interval);
    } catch (err) {
        console.error(err);
    }
});

// ── Poll game state ─────────────────────────────────────────────────────
async function pollState() {
    try {
        const response = await fetch('/state');
        const state = await response.json();

        // Dynamically adapt board size if server returns it
        if (state.board_size && state.board_size !== boardSize) {
            boardSize = state.board_size;
            initBoard();
        }

        if (state.board) {
            updateBoard(state.board);
        }

        if (state.human_colour && colorSelect) {
            colorSelect.value = state.human_colour;
        }

        currentTurn = state.turn;

        if (state.game_over) {
            gameIsOver = true;
            clearInterval(pollInterval);
            pollInterval = null;
            turnIndicator.textContent = `Game Over! Winner: ${state.winner || 'None'}`;
            statusPanel.className = 'status-panel';
            swapBtn.classList.add('hidden');
            return;
        }

        // AI-vs-AI: no waiting for human, just show "thinking"
        if (state.human_colour === null) {
            isHumanTurn = false;
            turnIndicator.textContent = `AI vs AI  —  Turn ${currentTurn}`;
            statusPanel.className = 'status-panel ai-turn';
        } else {
            isHumanTurn = state.waiting_for_human;
            updateStatusUI();
        }

        // Show swap button if it's turn 2 and human's turn
        if (isHumanTurn && currentTurn === 2) {
            swapBtn.classList.remove('hidden');
        } else {
            swapBtn.classList.add('hidden');
        }

    } catch (err) {
        console.error(err);
    }
}

function updateBoard(boardData) {
    const rows = boardData.length;
    const cols = rows > 0 ? boardData[0].length : 0;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const hex = document.querySelector(`.hex[data-r="${r}"][data-c="${c}"]`);
            if (hex) {
                const val = boardData[r][c];
                if (val === 'R') hex.className = 'hex red';
                else if (val === 'B') hex.className = 'hex blue';
                else hex.className = 'hex';
            }
        }
    }
}

function updateStatusUI() {
    if (gameIsOver) return;

    if (isHumanTurn) {
        turnIndicator.textContent = "Your Turn!";
        statusPanel.className = 'status-panel human-turn';
    } else {
        turnIndicator.textContent = "AI is thinking...";
        statusPanel.className = 'status-panel ai-turn';
    }
}

// ── Boot ─────────────────────────────────────────────────────────────────
loadConfig();
