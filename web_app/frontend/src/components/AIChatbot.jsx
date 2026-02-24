import React, { useState, useEffect, useRef } from 'react';
import './AIChatbot.css';
import { supabase } from '../supabaseClient';
import { useAuth } from '../context/AuthContext';

const AIChatbot = () => {
    const { token } = useAuth();
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { role: 'assistant', content: 'Welcome to AuraCare. I am your Clinical AI Assistant. How may I facilitate your healthcare journey today? I can help with appointment scheduling, accessibility settings, or general platform inquiries.' }
    ]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const [platformStats, setPlatformStats] = useState({ hospitals: 0, ambulances: 0 });
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        const fetchStats = async () => {
            const { count: hCount } = await supabase.from('hospitals').select('*', { count: 'exact', head: true });
            const { count: aCount } = await supabase.from('ambulances').select('*', { count: 'exact', head: true });
            setPlatformStats({ hospitals: hCount || 450, ambulances: aCount || 120 });
        };
        fetchStats();
    }, []);

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

        try {
            // Build headers - only include Authorization if we have a valid token
            const headers = {
                'Content-Type': 'application/json',
            };
            if (token && token !== 'null' && token !== 'undefined') {
                headers['Authorization'] = `Bearer ${token}`;
            }

            const response = await fetch('http://localhost:8000/api/v1/chat', {
                method: 'POST',
                headers,
                body: JSON.stringify({ message: input })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server responded with ${response.status}`);
            }

            const data = await response.json();
            setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
        } catch (error) {
            console.error('Chat error:', error);
            let errorMessage = 'I apologize, but I am currently experiencing connectivity issues with the Clinical Intelligence network. Please ensure the backend server is running and try again shortly.';

            if (error.message.includes('401') || error.message.includes('credentials')) {
                errorMessage = 'I was unable to verify your clinical credentials. Please try signing out and signing back in to refresh your session.';
            }

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: errorMessage
            }]);
        } finally {
            setIsTyping(false);
        }
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
