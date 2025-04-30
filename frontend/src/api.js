// API client for RecThink
const API_BASE_URL = 'http://localhost:8000/api';
const API_TIMEOUT = 60000; // 60 seconds timeout

// Helper function to add timeout to fetch requests
const fetchWithTimeout = async (url, options, timeout = API_TIMEOUT) => {
  const controller = new AbortController();
  const { signal } = controller;
  
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, { ...options, signal });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('Request timed out. The server may be overloaded or offline.');
    }
    throw error;
  }
};

export const initializeChat = async (apiKey, model, thinkingSystem = 'necort') => {
  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/initialize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        api_key: apiKey, 
        model,
        thinking_system: thinkingSystem 
      }),
    });
    
    if (!response.ok) {
      let errorText = `Failed to initialize chat: ${response.statusText}`;
      try {
        const errorData = await response.json();
        if (errorData && errorData.detail) {
          errorText = errorData.detail;
        }
      } catch (e) {
        // Ignore JSON parsing errors
      }
      throw new Error(errorText);
    }
    
    return response.json();
  } catch (error) {
    console.error("Initialize chat error:", error);
    throw error;
  }
};

export const sendMessage = async (sessionId, message, options = {}) => {
  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/send_message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: sessionId,
        message,
        thinking_rounds: options.thinkingRounds,
        alternatives_per_round: options.alternativesPerRound,
        thinking_system: options.thinkingSystem || 'necort'
      }),
    });
    
    if (!response.ok) {
      let errorText = `Failed to send message: ${response.statusText}`;
      try {
        const errorData = await response.json();
        if (errorData && errorData.detail) {
          errorText = errorData.detail;
        }
      } catch (e) {
        // Ignore JSON parsing errors
      }
      throw new Error(errorText);
    }
    
    return response.json();
  } catch (error) {
    console.error("Send message error:", error);
    throw error;
  }
};

export const saveConversation = async (sessionId, filename = null, fullLog = false) => {
  const response = await fetch(`${API_BASE_URL}/save`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      filename,
      full_log: fullLog,
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to save conversation: ${response.statusText}`);
  }
  
  return response.json();
};

export const listSessions = async () => {
  const response = await fetch(`${API_BASE_URL}/sessions`);
  
  if (!response.ok) {
    throw new Error(`Failed to list sessions: ${response.statusText}`);
  }
  
  return response.json();
};

export const deleteSession = async (sessionId) => {
  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    throw new Error(`Failed to delete session: ${response.statusText}`);
  }
  
  return response.json();
};

export const createWebSocketConnection = (sessionId) => {
  const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
  return ws;
};
