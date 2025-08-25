import React, { useRef, useEffect, useState } from 'react';
import { useChat } from '../hooks/useChat';
import { ChatWidgetProps } from './types';
import { ChatHeader } from './ChatHeader';
import { MessageBubble } from './MessageBubble';
import { ChatInput } from './ChatInput';
import { Loader2 } from 'lucide-react';

export const ChatWidget: React.FC<ChatWidgetProps> = ({
  apiUrl = 'http://localhost:8000/chat',
  theme = {},
  draggable = true,
  resizable = true,
  initialPosition = { x: window.innerWidth - 400 - 20, y: 20 },
  initialSize = { width: 400, height: 500 },
}) => {
  const {
    messages,
    isLoading,
    isMinimized,
    error,
    sendMessage,
    clearHistory,
    toggleMinimized,
  } = useChat(apiUrl);

  const [position, setPosition] = useState(initialPosition);
  const [size, setSize] = useState(initialSize);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const widgetRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages]);

  // Handle dragging
  const handleMouseDown = (e: React.MouseEvent) => {
    if (!draggable || isResizing) return;
    
    setIsDragging(true);
    setDragStart({
      x: e.clientX - position.x,
      y: e.clientY - position.y,
    });
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;

    const newX = Math.max(0, Math.min(window.innerWidth - size.width, e.clientX - dragStart.x));
    const newY = Math.max(0, Math.min(window.innerHeight - size.height, e.clientY - dragStart.y));

    setPosition({ x: newX, y: newY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setIsResizing(false);
  };

  // Handle resizing
  const handleResizeMouseDown = (e: React.MouseEvent) => {
    if (!resizable) return;
    
    e.stopPropagation();
    setIsResizing(true);
    setDragStart({
      x: e.clientX - size.width,
      y: e.clientY - size.height,
    });
  };

  const handleResizeMouseMove = (e: MouseEvent) => {
    if (!isResizing) return;

    const newWidth = Math.max(320, Math.min(600, e.clientX - position.x));
    const newHeight = Math.max(300, Math.min(800, e.clientY - position.y));

    setSize({ width: newWidth, height: newHeight });
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    } else if (isResizing) {
      document.addEventListener('mousemove', handleResizeMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mousemove', handleResizeMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, isResizing, dragStart, position, size]);

  const widgetStyle = {
    left: `${position.x}px`,
    top: `${position.y}px`,
    width: `${size.width}px`,
    height: isMinimized ? 'auto' : `${size.height}px`,
  };

  const themeVars = {
    '--chat-primary': theme.primary || '#2563EB',
    '--chat-secondary': theme.secondary || '#F3F4F6',
    '--chat-background': theme.background || '#FFFFFF',
    '--chat-text': theme.text || '#1F2937',
  } as React.CSSProperties;

  return (
    <div
      ref={widgetRef}
      className={`fixed z-50 bg-white rounded-lg shadow-xl border border-gray-200 overflow-hidden transition-all duration-200 ${
        isDragging ? 'cursor-grabbing' : ''
      } ${isResizing ? 'cursor-nw-resize' : ''}`}
      style={{ ...widgetStyle, ...themeVars }}
    >
      <div onMouseDown={handleMouseDown}>
        <ChatHeader
          isMinimized={isMinimized}
          onToggleMinimize={toggleMinimized}
          onClearHistory={clearHistory}
        />
      </div>

      {!isMinimized && (
        <>
          <div
            ref={messagesRef}
            className="flex-1 overflow-y-auto p-4 space-y-2 bg-gray-50"
            style={{ height: `${size.height - 120}px` }}
          >
            {messages.length === 0 ? (
              <div className="text-center text-gray-500 mt-8">
                <p className="text-sm">👋 Hi! I'm here to help you find the perfect sound and lighting equipment for your event. What are you planning?</p>
              </div>
            ) : (
              messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))
            )}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200">
                  <div className="flex items-center space-x-2 text-gray-600">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">Searching our inventory...</span>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-3 py-2 rounded-lg text-sm">
                {error}
              </div>
            )}
          </div>

          <ChatInput
            onSendMessage={sendMessage}
            disabled={isLoading}
            placeholder={isLoading ? "AI is typing..." : "Ask about speakers, lighting, or DJ equipment..."}
          />

          {resizable && (
            <div
              className="absolute bottom-0 right-0 w-4 h-4 cursor-nw-resize opacity-0 hover:opacity-50 bg-gray-400 transition-opacity"
              onMouseDown={handleResizeMouseDown}
            />
          )}
        </>
      )}
    </div>
  );
};
