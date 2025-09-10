import React from 'react';
import { Message } from './types';
import { Bot, User, ExternalLink } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface MessageBubbleProps {
  message: Message;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const isBot = message.isBot;

  const formatMessage = (text: string) => {
    const splitRegex = /(https?:\/\/[^\s]+)/g;
    const testRegex = /^https?:\/\/[^\s]+$/i;
    const parts = text.split(splitRegex);

    return parts.map((part, index) => {
      if (testRegex.test(part)) {
        return (
          <a
            key={index}
            href={part}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center text-blue-600 hover:text-blue-800 underline"
          >
            View Product <ExternalLink className="w-3 h-3 ml-1" />
          </a>
        );
      }
      return <ReactMarkdown key={index}>{part}</ReactMarkdown>;
    });
  };

  return (
    <div
      className={`flex items-start gap-2 mb-4 ${
        isBot ? 'justify-start' : 'justify-end'
      }`}
    >
      {isBot && (
        <div className="flex-shrink-0 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
          <Bot className="w-4 h-4 text-white" />
        </div>
      )}
      
      <div
        className={`max-w-xs lg:max-w-md px-3 py-2 rounded-lg ${
          isBot
            ? 'bg-white text-gray-800 border border-gray-200 shadow-sm'
            : 'bg-blue-600 text-white'
        } ${message.isStreaming ? 'animate-pulse' : ''}`}
      >
        <div className="text-sm whitespace-pre-wrap break-words">
          {isBot ? formatMessage(message.text) : message.text}
          {message.isStreaming && (
            <span className="inline-block w-2 h-4 bg-current opacity-75 animate-pulse ml-1" />
          )}
        </div>
        <p className={`text-xs mt-1 opacity-70`}>
          {formatTime(message.timestamp)}
        </p>
      </div>

      {!isBot && (
        <div className="flex-shrink-0 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
          <User className="w-4 h-4 text-white" />
        </div>
      )}
    </div>
  );
};
