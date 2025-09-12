export interface Message {
  id: string;
  text: string;
  isBot: boolean;
  timestamp: number;
  isStreaming?: boolean;
}

export interface ChatWidgetProps {
  apiUrl?: string;
  theme?: {
    primary?: string;
    secondary?: string;
    background?: string;
    text?: string;
  };
  draggable?: boolean;
  resizable?: boolean;
  initialPosition?: { x: number; y: number };
  initialSize?: { width: number; height: number };
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  isMinimized: boolean;
  error: string | null;
}
