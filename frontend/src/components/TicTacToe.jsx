import React, { useState, useEffect } from 'react';
import { useRecThink } from '../context/RecThinkContext';
import { Brain, RefreshCw } from 'lucide-react';

const TicTacToe = () => {
  const { 
    apiKey, 
    model, 
    thinkingSystem,
    isThinking,
    sessionId,
    initializeChat,
    sendMessage
  } = useRecThink();
  
  const [board, setBoard] = useState(Array(9).fill(''));
  const [currentPlayer, setCurrentPlayer] = useState('X'); // X is user, O is AI
  const [gameStatus, setGameStatus] = useState('ready'); // ready, playing, won, draw
  const [winner, setWinner] = useState(null);
  const [aiThinking, setAiThinking] = useState(false);
  const [gameHistory, setGameHistory] = useState([]);
  
  // Initialize game
  const startGame = () => {
    setBoard(Array(9).fill(''));
    setCurrentPlayer('X');
    setGameStatus('playing');
    setWinner(null);
    setGameHistory([]);
  };
  
  // Check for winner
  const checkWinner = (board) => {
    const lines = [
      [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
      [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
      [0, 4, 8], [2, 4, 6]             // diagonals
    ];
    
    for (let i = 0; i < lines.length; i++) {
      const [a, b, c] = lines[i];
      if (board[a] && board[a] === board[b] && board[a] === board[c]) {
        return board[a];
      }
    }
    
    return null;
  };
  
  // Check for draw
  const checkDraw = (board) => {
    return board.every(cell => cell !== '');
  };
  
  // Make a move
  const handleCellClick = (index) => {
    if (!apiKey) {
      alert("Please set your API key in settings first");
      return;
    }
    
    if (board[index] !== '' || gameStatus !== 'playing' || currentPlayer !== 'X' || aiThinking) {
      return;
    }
    
    // Update board with user's move
    const newBoard = [...board];
    newBoard[index] = 'X';
    setBoard(newBoard);
    
    // Create a new history array with the player's move
    const updatedHistory = [...gameHistory, { player: 'X', position: index }];
    setGameHistory(updatedHistory);
    
    // Check for win or draw
    const playerWon = checkWinner(newBoard);
    const isDraw = checkDraw(newBoard);
    
    if (playerWon) {
      setGameStatus('won');
      setWinner('X');
      return;
    }
    
    if (isDraw) {
      setGameStatus('draw');
      return;
    }
    
    // Switch to AI's turn
    setCurrentPlayer('O');
    getAIMove(newBoard);
  };
  
  // Get AI move using recursive thinking
  const getAIMove = async (currentBoard) => {
    try {
      setAiThinking(true);
      
      // Ensure session is initialized
      if (!sessionId) {
        try {
          await initializeChat();
          // If initialization fails, we'll catch it below
        } catch (initError) {
          console.error("Failed to initialize chat:", initError);
          alert("Failed to connect to AI. Please check your API key in settings.");
          setGameStatus('ready');
          setAiThinking(false);
          return;
        }
      }
      
      // Format the board for the AI
      const gameState = {
        board: [
          [currentBoard[0] || ' ', currentBoard[1] || ' ', currentBoard[2] || ' '],
          [currentBoard[3] || ' ', currentBoard[4] || ' ', currentBoard[5] || ' '],
          [currentBoard[6] || ' ', currentBoard[7] || ' ', currentBoard[8] || ' ']
        ],
        history: gameHistory
      };
      
      // Ask AI for the next move using the chosen thinking system
      const prompt = `I'm playing Tic Tac Toe. You are playing as O. Here's the current board state (spaces are empty cells):
      
${gameState.board[0][0]}|${gameState.board[0][1]}|${gameState.board[0][2]}
-+-+-
${gameState.board[1][0]}|${gameState.board[1][1]}|${gameState.board[1][2]}
-+-+-
${gameState.board[2][0]}|${gameState.board[2][1]}|${gameState.board[2][2]}

The positions are numbered as follows:
0|1|2
3|4|5
6|7|8

Based on this board state, what is your next move? Answer with only a single number 0-8.`;

      let result;
      try {
        result = await sendMessage(prompt);
      } catch (sendError) {
        console.error("Failed to get AI response:", sendError);
        alert("Failed to get AI move. The server might be down or your API key may be invalid.");
        setCurrentPlayer('X');
        setAiThinking(false);
        return;
      }
      
      if (!result || !result.response) {
        console.error("Received empty response from AI");
        setCurrentPlayer('X');
        setAiThinking(false);
        return;
      }
      
      // Extract move from AI response
      let aiMoveIndex = -1;
      const responseText = result.response;
      
      // Try different regex patterns to extract a valid move
      const exactNumberPattern = /^\s*([0-8])\s*$/;
      const numbersInResponse = responseText.match(/\b[0-8]\b/g);
      
      // First try to find a standalone number
      const exactMatch = responseText.match(exactNumberPattern);
      if (exactMatch) {
        aiMoveIndex = parseInt(exactMatch[1], 10);
      } 
      // Then try to find any number 0-8 in the response
      else if (numbersInResponse && numbersInResponse.length > 0) {
        aiMoveIndex = parseInt(numbersInResponse[0], 10);
      }
      
      console.log("AI response:", responseText);
      console.log("Extracted move:", aiMoveIndex);
      
      // Validate the move
      if (aiMoveIndex >= 0 && aiMoveIndex <= 8 && currentBoard[aiMoveIndex] === '') {
        // Valid move, update the board
        const newBoard = [...currentBoard];
        newBoard[aiMoveIndex] = 'O';
        setBoard(newBoard);
        
        // Create a new history array with the AI's move
        const updatedHistory = [...gameHistory, { player: 'O', position: aiMoveIndex }];
        setGameHistory(updatedHistory);
        
        // Check for win or draw
        const aiWon = checkWinner(newBoard);
        const isDraw = checkDraw(newBoard);
        
        if (aiWon) {
          setGameStatus('won');
          setWinner('O');
        } else if (isDraw) {
          setGameStatus('draw');
        } else {
          // Back to player's turn
          setCurrentPlayer('X');
        }
      } else {
        // Invalid move, try to find an empty cell
        console.error("AI returned invalid move:", aiMoveIndex, "for board:", currentBoard);
        
        // Find any available empty cell as fallback
        const emptyCells = currentBoard.map((cell, idx) => cell === '' ? idx : -1).filter(idx => idx !== -1);
        
        if (emptyCells.length > 0) {
          // Take the first available empty cell
          const fallbackMove = emptyCells[0];
          const newBoard = [...currentBoard];
          newBoard[fallbackMove] = 'O';
          setBoard(newBoard);
          
          // Create a new history array with the fallback move
          const updatedHistory = [...gameHistory, { player: 'O', position: fallbackMove }];
          setGameHistory(updatedHistory);
          
          console.log("Using fallback move:", fallbackMove);
          
          // Check for win or draw
          const aiWon = checkWinner(newBoard);
          const isDraw = checkDraw(newBoard);
          
          if (aiWon) {
            setGameStatus('won');
            setWinner('O');
          } else if (isDraw) {
            setGameStatus('draw');
          } else {
            setCurrentPlayer('X');
          }
        } else {
          // No empty cells left, it's a draw
          setGameStatus('draw');
        }
      }
    } catch (error) {
      console.error("Error in AI move processing:", error);
      alert("An error occurred while processing the AI move. Please try again.");
      // Let player try again
      setCurrentPlayer('X');
    } finally {
      setAiThinking(false);
    }
  };
  
  // Game status message
  const getStatusMessage = () => {
    if (!apiKey) return "Set your API key in settings to play";
    if (gameStatus === 'ready') return "Press Start to play";
    if (gameStatus === 'playing' && currentPlayer === 'X') return "Your turn";
    if (gameStatus === 'playing' && currentPlayer === 'O') return "AI is thinking...";
    if (gameStatus === 'won') return `${winner === 'X' ? 'You' : 'AI'} won!`;
    if (gameStatus === 'draw') return "It's a draw!";
    return "";
  };
  
  // Highlight winning cells
  const isWinningCell = (index) => {
    if (gameStatus !== 'won' || !winner) return false;
    
    const lines = [
      [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
      [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
      [0, 4, 8], [2, 4, 6]             // diagonals
    ];
    
    for (let i = 0; i < lines.length; i++) {
      const [a, b, c] = lines[i];
      if (board[a] && board[a] === board[b] && board[a] === board[c] && board[a] === winner) {
        return lines[i].includes(index);
      }
    }
    
    return false;
  };
  
  // Render a single cell
  const renderCell = (index) => {
    const winningCell = isWinningCell(index);
    
    return (
      <div 
        key={index}
        className={`w-20 h-20 border-2 flex items-center justify-center text-4xl font-bold cursor-pointer
          ${winningCell ? 'border-green-500 bg-green-50' : 'border-gray-300'}
          ${board[index] === '' && gameStatus === 'playing' && currentPlayer === 'X' && !aiThinking ? 'hover:bg-blue-100' : ''}
          ${board[index] === 'X' ? 'text-blue-600' : 'text-red-600'}`}
        onClick={() => handleCellClick(index)}
      >
        {board[index]}
      </div>
    );
  };
  
  // Render the board
  const renderBoard = () => {
    return (
      <div className="grid grid-cols-3 gap-1 my-8">
        {[...Array(9)].map((_, index) => renderCell(index))}
      </div>
    );
  };
  
  // Display winner information
  const renderWinnerInfo = () => {
    if (!winner || gameStatus !== 'won') return null;
    
    return (
      <div className={`mt-4 p-4 rounded-lg text-center ${winner === 'X' ? 'bg-blue-100 text-blue-700' : 'bg-red-100 text-red-700'}`}>
        <h3 className="text-lg font-bold">
          {winner === 'X' ? 'You won!' : 'AI won!'}
        </h3>
        <p className="mt-1 text-sm">
          {winner === 'X' 
            ? 'Congratulations! You beat the AI.' 
            : `The ${thinkingSystem === 'necort' ? 'NECoRT' : 'CoRT'} AI outplayed you.`}
        </p>
      </div>
    );
  };
  
  return (
    <div className="flex flex-col items-center p-8">
      <h2 className="text-2xl font-bold mb-4">Tic Tac Toe</h2>
      
      <div className="mb-4 flex items-center">
        <div className="mr-4">
          <span className="text-gray-700">Thinking System:</span>
          <span className="ml-2 bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full flex items-center">
            <Brain size={12} className="mr-1" />
            {thinkingSystem === 'necort' ? 'NECoRT' : 'CoRT'}
          </span>
        </div>
        <div className="mr-4">
          <span className="text-gray-700">Model:</span>
          <span className="ml-2 bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded-full">
            {model.split('/')[1]}
          </span>
        </div>
        <div>
          <span className={`inline-block w-2 h-2 mr-1 rounded-full ${
            sessionId ? 'bg-green-500' : 'bg-gray-400'
          }`}></span>
          <span className="text-xs text-gray-600">
            {sessionId ? 'Connected' : 'Not connected'}
          </span>
        </div>
      </div>
      
      {!apiKey && (
        <div className="mb-4 p-4 bg-yellow-50 text-yellow-700 rounded-lg max-w-lg text-center">
          <p className="font-medium">API Key Required</p>
          <p className="text-sm mt-1">Go to the Settings tab to enter your OpenRouter API key</p>
        </div>
      )}
      
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="text-center mb-4">
          <span className="text-lg font-medium">
            {getStatusMessage()}
          </span>
          {aiThinking && (
            <div className="mt-2 text-sm text-blue-600 flex items-center justify-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
              AI is thinking with {thinkingSystem === 'necort' ? 'Nash Equilibrium' : 'standard'} recursive thinking...
            </div>
          )}
        </div>
        
        {renderBoard()}
        
        {renderWinnerInfo()}
        
        <div className="flex justify-center">
          <button
            className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded flex items-center"
            onClick={startGame}
            disabled={!apiKey || aiThinking}
          >
            <RefreshCw size={16} className="mr-2" />
            {gameStatus === 'ready' ? 'Start Game' : 'Restart Game'}
          </button>
        </div>
      </div>
      
      {gameStatus === 'won' && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg max-w-lg">
          <h3 className="text-lg font-medium mb-2">Game Stats</h3>
          <p className="text-sm">
            <strong>Winner:</strong> {winner === 'X' ? 'Human (X)' : 'AI (O)'}
          </p>
          <p className="text-sm">
            <strong>Thinking System:</strong> {thinkingSystem === 'necort' ? 'NECoRT' : 'CoRT'}
          </p>
          <p className="text-sm">
            <strong>Model:</strong> {model.split('/')[1]}
          </p>
        </div>
      )}
    </div>
  );
};

export default TicTacToe; 