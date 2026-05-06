const boardSize = 11;
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

let pollInterval = null;
let currentTurn = 0;
let isHumanTurn = false;
let gameIsOver = false;

// Update temperature display
tempSlider.addEventListener('input', (e) => {
    tempVal.textContent = e.target.value;
});

// Initialize the board UI
function initBoard() {
    boardEl.innerHTML = '';
    
    // Calculate board dimensions
    const totalWidth = hexWidth * boardSize + (boardSize * hexWidth / 2);
    const totalHeight = hexHeight * 0.75 * boardSize + hexHeight * 0.25;
    
    boardEl.style.width = `${totalWidth}px`;
    boardEl.style.height = `${totalHeight}px`;

    // Add visual sides
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
            
            // Pointy topped hex positioning
            const x = c * hexWidth + (r * hexWidth / 2);
            const y = r * hexHeight * 0.75;
            
            hex.style.left = `${x}px`;
            hex.style.top = `${y}px`;
            
            hex.addEventListener('click', () => onHexClick(r, c));
            boardEl.appendChild(hex);
        }
    }
}

// Handle Hex click
async function onHexClick(r, c) {
    if (!isHumanTurn || gameIsOver) return;
    
    try {
        const response = await fetch('/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x: r, y: c })
        });
        
        if (response.ok) {
            // Optimistically update
            const hex = document.querySelector(`.hex[data-r="${r}"][data-c="${c}"]`);
            if (hex) {
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

// Handle Swap Rule
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

// Start Game
startBtn.addEventListener('click', async () => {
    initBoard();
    gameIsOver = false;
    
    const humanColor = colorSelect.value;
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
        pollInterval = setInterval(pollState, 500);
    } catch (err) {
        console.error(err);
    }
});

// Poll game state
async function pollState() {
    try {
        const response = await fetch('/state');
        const state = await response.json();
        
        if (state.board) {
            updateBoard(state.board);
        }
        
        if (state.human_colour) {
            colorSelect.value = state.human_colour;
        }
        
        currentTurn = state.turn;
        
        if (state.game_over) {
            gameIsOver = true;
            clearInterval(pollInterval);
            pollInterval = null;
            turnIndicator.textContent = `Game Over! Winner: ${state.winner}`;
            statusPanel.className = 'status-panel';
            swapBtn.classList.add('hidden');
            return;
        }
        
        isHumanTurn = state.waiting_for_human;
        updateStatusUI();
        
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
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
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

// Initial render
initBoard();
