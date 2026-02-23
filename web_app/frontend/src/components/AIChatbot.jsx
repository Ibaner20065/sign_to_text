import React, { useState, useEffect, useRef } from 'react';
import './AIChatbot.css';

const AIChatbot = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { role: 'assistant', content: 'Welcome to AuraCare. I am your Clinical AI Assistant. How may I facilitate your healthcare journey today? I can help with appointment scheduling, accessibility settings, or general platform inquiries.' }
    ]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        if (isOpen) {
            scrollToBottom();
        }
    }, [messages, isOpen]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsTyping(true);

        // Simulated AI Response with formal medical tone
        setTimeout(() => {
            let botContent = "I have received your inquiry. As an AI assistant, I can confirm that our platform supports full HIPAA-compliant clinical workflows. Would you like me to direct you to our Physician Discovery portal or the Emergency Services dispatch?";

            if (input.toLowerCase().includes('ambulance')) {
                botContent = "Emergency Services protocol initiated. You can access immediate dispatch via our Emergency Banner or by navigating to the /ambulance dashboard. No authentication is required for critical care.";
            } else if (input.toLowerCase().includes('sign') || input.toLowerCase().includes('deaf')) {
                botContent = "AuraCare uses proprietary Neural Sign Recognition (NSR) technology. Our system translates sign language to clinical-grade speech in real-time with <200ms latency. You can test this in the Communication Dashboard.";
            } else if (input.toLowerCase().includes('hospital') || input.toLowerCase().includes('bed')) {
                botContent = "I can assist with facility reservations. We currently monitor bed availability across 450+ accredited medical institutions. Would you like to check current capacity?";
            }

            setMessages(prev => [...prev, { role: 'assistant', content: botContent }]);
            setIsTyping(false);
        }, 1500);
    };

    return (
        <div className={`ai-chatbot-container ${isOpen ? 'open' : ''}`}>
            {/* Toggle Button */}
            <button
                className="ai-chatbot-toggle"
                onClick={() => setIsOpen(!isOpen)}
                aria-label="Toggle Clinical AI Assistant"
            >
                {isOpen ? (
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                ) : (
                    <div className="ai-chatbot-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                        <span className="ai-chatbot-status"></span>
                    </div>
                )}
            </button>

            {/* Chat Window */}
            {isOpen && (
                <div className="ai-chatbot-window">
                    <div className="ai-chatbot-header">
                        <div className="ai-chatbot-title">
                            <span className="bio-glow-dot"></span>
                            <div>
                                <h4>AuraCare Assistant</h4>
                                <p>Clinical Intelligence v4.2</p>
                            </div>
                        </div>
                        <div className="ai-chatbot-controls">
                            <span className="badge-hipaa">HIPAA</span>
                        </div>
                    </div>

                    <div className="ai-chatbot-messages">
                        {messages.map((msg, i) => (
                            <div key={i} className={`chat-message ${msg.role}`}>
                                <div className="message-content">
                                    {msg.content}
                                </div>
                            </div>
                        ))}
                        {isTyping && (
                            <div className="chat-message assistant typing">
                                <div className="message-content">
                                    <span className="typing-dot"></span>
                                    <span className="typing-dot"></span>
                                    <span className="typing-dot"></span>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    <form className="ai-chatbot-input" onSubmit={handleSend}>
                        <input
                            type="text"
                            placeholder="Inquire about clinical services..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                        />
                        <button type="submit" disabled={!input.trim()}>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                        </button>
                    </form>
                </div>
            )}
        </div>
    );
};

export default AIChatbot;
