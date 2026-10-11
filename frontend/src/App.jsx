import { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Loader2 } from 'lucide-react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I have read your PDFs. What would you like to know?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      // Connect exactly to our FastAPI endpoint
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMessage })
      });

      if (!response.ok) throw new Error("API failed");
      const data = await response.json();

      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer,
        sources: data.sources // The array of filenames we extracted in FastAPI!
      }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', content: "Error: Could not connect to the backend. Is FastAPI running on port 8000?" }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        Corporate PDF Analyst (RAG)
      </header>
      
      <main className="chat-box">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message-wrapper ${msg.role}`}>
            <div className="icon">
              {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
            </div>
            <div className="message-content">
              <p>{msg.content}</p>
              
              {/* If we have sources from ChromaDB, render them nicely! */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources-badge">
                  <strong>Retrieved from: </strong>
                  {msg.sources.map((source, i) => (
                    <span key={i} className="badge">{source.split(/[\\/]/).pop()}</span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        
        {/* Loading state indicator */}
        {isLoading && (
          <div className="message-wrapper assistant">
            <div className="icon"><Loader2 className="spinner" size={20} /></div>
            <div className="message-content"><p>Reading documents...</p></div>
          </div>
        )}
        <div ref={chatEndRef} />
      </main>

      <form className="input-area" onSubmit={sendMessage}>
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about your documents..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          <Send size={20} />
        </button>
      </form>
    </div>
  );
}

export default App;
