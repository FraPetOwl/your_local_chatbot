import { useState, useCallback, useEffect, useRef } from 'react';
import { Message, ChatState } from '../Components/types';

const STORAGE_KEY = 'superior-sounds-chatbot-messages';

export const useChat = (apiUrl = 'http://localhost:8000/chat') => {
  const [state, setState] = useState<ChatState>({
    messages: [],
    isLoading: false,
    isMinimized: true,
    error: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  // Load messages from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const messages = JSON.parse(saved);
        setState(prev => ({ ...prev, messages }));
      } catch (error) {
        console.error('Failed to load chat history:', error);
      }
    }
  }, []);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (state.messages.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.messages));
    }
  }, [state.messages]);

  const addMessage = useCallback((message: Omit<Message, 'id'>) => {
    const newMessage: Message = {
      ...message,
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
    };

    setState(prev => ({
      ...prev,
      messages: [...prev.messages, newMessage],
    }));

    return newMessage.id;
  }, []);

  const updateMessage = useCallback((id: string, updates: Partial<Message>) => {
    setState(prev => ({
      ...prev,
      messages: prev.messages.map(msg =>
        msg.id === id ? { ...msg, ...updates } : msg
      ),
    }));
  }, []);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || state.isLoading) return;

    // Add user message
    addMessage({
      text: text.trim(),
      isBot: false,
      timestamp: Date.now(),
    });

    // Add streaming bot message
    const botMessageId = addMessage({
      text: '',
      isBot: true,
      timestamp: Date.now(),
      isStreaming: true,
    });

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      // Cancel any previous request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      abortControllerRef.current = new AbortController();

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: text.trim(),
          style: 'concise',
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Response body is not readable');
      }

      let accumulatedText = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = new TextDecoder().decode(value);
        accumulatedText += chunk;

        updateMessage(botMessageId, {
          text: accumulatedText,
          isStreaming: true,
        });
      }

      // Mark streaming as complete
      updateMessage(botMessageId, {
        text: accumulatedText,
        isStreaming: false,
      });

    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        return; // Request was cancelled
      }

      console.error('Chat API error:', error);
      
      updateMessage(botMessageId, {
        text: 'Sorry, I encountered an error. Please try again.',
        isStreaming: false,
      });

      setState(prev => ({
        ...prev,
        error: 'Failed to send message. Please check your connection and try again.',
      }));
    } finally {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  }, [apiUrl, state.isLoading, addMessage, updateMessage]);

  const clearHistory = useCallback(() => {
    setState(prev => ({ ...prev, messages: [] }));
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  const toggleMinimized = useCallback(() => {
    setState(prev => ({ ...prev, isMinimized: !prev.isMinimized }));
  }, []);

  return {
    ...state,
    sendMessage,
    clearHistory,
    toggleMinimized,
  };
};
