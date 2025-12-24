import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, FileText, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: '你好！我是医保政策智能助手。请问有什么关于医保报销、待遇政策的问题可以帮您？' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Placeholder for assistant message
    const assistantMessageId = Date.now();
    setMessages(prev => [...prev, { 
      role: 'assistant', 
      content: '', 
      contexts: [], 
      id: assistantMessageId,
      isStreaming: true 
    }]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage.content })
      });

      if (!response.ok) throw new Error('Network response was not ok');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const json = JSON.parse(line);
            
            setMessages(prev => prev.map(msg => {
              if (msg.id !== assistantMessageId) return msg;
              
              if (json.type === 'contexts') {
                return { ...msg, contexts: json.data };
              } else if (json.type === 'chunk') {
                return { ...msg, content: msg.content + json.data };
              }
              return msg;
            }));
          } catch (e) {
            console.error('Error parsing JSON line:', e);
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => prev.map(msg => {
        if (msg.id !== assistantMessageId) return msg;
        return { ...msg, content: '抱歉，系统出现错误，请稍后再试。' };
      }));
    } finally {
      setIsLoading(false);
      setMessages(prev => prev.map(msg => {
        if (msg.id !== assistantMessageId) return msg;
        return { ...msg, isStreaming: false };
      }));
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto bg-white shadow-xl">
      {/* Header */}
      <header className="bg-blue-600 text-white p-4 shadow-md flex items-center gap-2">
        <Bot className="w-8 h-8" />
        <h1 className="text-xl font-bold">医保政策智能问答</h1>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-gray-50">
        {messages.map((msg, idx) => (
          <div key={idx} className={cn(
            "flex gap-4 max-w-3xl",
            msg.role === 'user' ? "ml-auto flex-row-reverse" : ""
          )}>
            <div className={cn(
              "w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0",
              msg.role === 'user' ? "bg-blue-500 text-white" : "bg-green-500 text-white"
            )}>
              {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
            </div>
            
            <div className="flex flex-col gap-2 max-w-[80%]">
              <div className={cn(
                "p-4 rounded-2xl shadow-sm prose prose-sm max-w-none",
                msg.role === 'user' 
                  ? "bg-blue-600 text-white rounded-tr-none prose-invert" 
                  : "bg-white text-gray-800 rounded-tl-none border border-gray-200"
              )}>
                <ReactMarkdown>{msg.content}</ReactMarkdown>
                {msg.isStreaming && <span className="inline-block w-2 h-4 ml-1 bg-gray-400 animate-pulse"/>}
              </div>

              {/* Contexts Reference */}
              {msg.contexts && msg.contexts.length > 0 && (
                <div className="bg-yellow-50 border border-yellow-100 rounded-lg p-3 text-xs text-gray-600">
                  <div className="font-semibold flex items-center gap-1 mb-2 text-yellow-700">
                    <FileText size={14} /> 参考来源
                  </div>
                  <div className="space-y-2">
                    {msg.contexts.map((ctx, i) => (
                      <details key={i} className="group cursor-pointer">
                        <summary className="list-none hover:text-blue-600 flex gap-2">
                          <span className="font-mono text-yellow-600">[{ctx.rank}]</span>
                          <span className="truncate flex-1">{ctx.filename} (p{ctx.page})</span>
                          <span className="text-gray-400 text-[10px]">{(ctx.score * 100).toFixed(1)}%</span>
                        </summary>
                        <div className="mt-1 pl-6 p-2 bg-white rounded border border-gray-100 text-gray-500 italic">
                          "{ctx.text}"
                        </div>
                      </details>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <form onSubmit={handleSubmit} className="p-4 bg-white border-t border-gray-200">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="请输入您的问题，例如：哈尔滨退休人员住院报销比例是多少？"
            className="flex-1 p-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-blue-600 text-white p-3 rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {isLoading ? <Loader2 className="animate-spin" /> : <Send size={20} />}
            <span className="hidden sm:inline">发送</span>
          </button>
        </div>
      </form>
    </div>
  );
}

export default App;
