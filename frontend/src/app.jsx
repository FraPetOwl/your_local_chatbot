import React from 'react';
import { ChatWidget } from './components_tmp/ChatWidget';
console.log(">>> Rendering ChatWidget component");


function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-8">
          Your React Application
        </h1>
        
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-700 mb-4">
            Main Content Area
          </h2>
          <p className="text-gray-600 mb-4">
            This is your main application content. The chatbot widget is floating 
            independently and can be moved around, resized, and minimized.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-medium text-gray-800 mb-2">Feature 1</h3>
              <p className="text-gray-600 text-sm">
                Your application features go here. The chat widget won't interfere 
                with your existing layout.
              </p>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-medium text-gray-800 mb-2">Feature 2</h3>
              <p className="text-gray-600 text-sm">
                The widget maintains its position and chat history across 
                page reloads and navigation.
              </p>
            </div>
          </div>
        </div>
      </div>

      <ChatWidget
        apiUrl="http://158.101.102.126:8000"
        draggable={true}
        resizable={true}
        theme={{
          primary: '#3B82F6',
          secondary: '#F3F4F6',
          background: '#FFFFFF',
          text: '#1F2937',
        }}
      />
    </div>
  );
}

export default App;