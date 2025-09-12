import React from 'react';
import { Minimize2, Maximize2, Trash2, MessageCircle } from 'lucide-react';

interface ChatHeaderProps {
  isMinimized: boolean;
  onToggleMinimize: () => void;
  onClearHistory: () => void;
  title?: string;
}

export const ChatHeader: React.FC<ChatHeaderProps> = ({
  isMinimized,
  onToggleMinimize,
  onClearHistory,
  title = "Superior Sounds",
}) => {
  return (
    <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-3 rounded-t-lg flex items-center justify-between cursor-move">
      <div className="flex items-center gap-2">
        <MessageCircle className="w-5 h-5" />
        <div>
          <h3 className="font-medium text-sm">{title}</h3>
          <p className="text-xs opacity-90">Equipment Rental Assistant</p>
        </div>
      </div>
      
      <div className="flex items-center gap-1">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onClearHistory();
          }}
          className="p-1 hover:bg-blue-800 rounded transition-colors"
          title="Clear chat history"
        >
          <Trash2 className="w-4 h-4" />
        </button>
        
        <button
          onClick={(e) => {
            e.stopPropagation();
            onToggleMinimize();
          }}
          className="p-1 hover:bg-blue-800 rounded transition-colors"
          title={isMinimized ? "Maximize" : "Minimize"}
        >
          {isMinimized ? (
            <Maximize2 className="w-4 h-4" />
          ) : (
            <Minimize2 className="w-4 h-4" />
          )}
        </button>
      </div>
    </div>
  );
};
