"use client";

import { Message } from '@/lib/types';
import { ArrowUp, Search, Loader2, Sparkles, Paperclip, X, FileUp } from 'lucide-react';
import { RefObject, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { CodeBlock } from '@/components/blocks/CodeBlock';

const ACCEPT_FILES = '.pdf,.txt,.md,.html,.htm,.docx';
const MAX_FILE_SIZE_MB = 20;

export interface ChatPanelProps {
    messages: Message[];
    input: string;
    setInput: (v: string) => void;
    onSend: (e: React.FormEvent) => void;
    isWebMode: boolean;
    setIsWebMode: (v: boolean) => void;
    isResearchMode: boolean;
    setIsResearchMode: (v: boolean) => void;
    isStreaming: boolean;
    placeholder?: string;
    showResearchToggle?: boolean;
    className?: string;
    messagesEndRef?: RefObject<HTMLDivElement | null>;
    /** Attached files (e.g. for this message). Shown as chips; parent handles upload on send. */
    attachedFiles?: File[];
    onAttachedFilesChange?: (files: File[]) => void;
    /** Optional button/trigger for options (model, research params). Rendered next to mode toggles. */
    optionsTrigger?: React.ReactNode;
}

export function ChatPanel({
    messages,
    input,
    setInput,
    onSend,
    isWebMode,
    setIsWebMode,
    isResearchMode,
    setIsResearchMode,
    isStreaming,
    placeholder = "Message Wort…",
    showResearchToggle = true,
    className = "",
    messagesEndRef,
    attachedFiles = [],
    onAttachedFilesChange,
    optionsTrigger,
}: ChatPanelProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const list = e.target.files ? Array.from(e.target.files) : [];
        if (!list.length || !onAttachedFilesChange) return;
        const valid = list.filter((f) => f.size <= MAX_FILE_SIZE_MB * 1024 * 1024);
        onAttachedFilesChange([...attachedFiles, ...valid].slice(0, 5));
        e.target.value = '';
    };

    const removeFile = (index: number) => {
        onAttachedFilesChange?.(attachedFiles.filter((_, i) => i !== index));
    };

    return (
        <div className={`flex flex-col h-full min-h-0 ${className}`}>
            <div className="flex-1 min-h-0 overflow-y-auto scroll-smooth pb-8">
                {messages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center min-h-[280px] text-center px-6 py-8">
                        <div className="w-16 h-16 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mb-6">
                            <Sparkles className="w-8 h-8 text-primary" aria-hidden />
                        </div>
                        <h2 className="text-xl font-semibold text-foreground mb-3 tracking-tight">Start a conversation</h2>
                        <p className="text-[15px] text-muted-foreground max-w-sm leading-[1.75]">
                            Ask anything. Attach files, use web search, or start deep research — all from here.
                        </p>
                    </div>
                ) : (
                    <div className="space-y-10 w-full max-w-4xl mx-auto px-5 pt-6">
                        {messages.map((msg) => (
                            <div
                                key={msg.id}
                                className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                            >
                                {msg.role === 'assistant' && (
                                    <div className="w-9 h-9 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0 mt-1.5">
                                        <Sparkles className="w-5 h-5 text-primary" aria-hidden />
                                    </div>
                                )}
                                <div
                                    className={`flex-1 min-w-0 rounded-2xl px-5 py-4 ${
                                        msg.role === 'user'
                                            ? 'rounded-br-md bg-primary/10 border border-primary/20 text-foreground'
                                            : 'rounded-bl-md bg-card border border-border'
                                    }`}
                                >
                                    {msg.role === 'assistant' && msg.mode === 'web' && (
                                        <span className="inline-flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-primary/90 mb-3">
                                            <Search className="w-3.5 h-3.5" aria-hidden /> Web
                                        </span>
                                    )}
                                    <div
                                        className={`prose prose-sm max-w-none prose-invert prose-readable prose-p:leading-[1.95] prose-p:my-[1.1em] prose-headings:my-3 prose-pre:my-4 prose-pre:bg-white/5 prose-pre:border prose-pre:border-white/10 prose-pre:rounded-xl prose-pre:px-4 prose-pre:py-3 ${
                                            msg.role === 'user' ? 'text-foreground text-[15px]' : 'text-foreground/95 text-[15px]'
                                        }`}
                                    >
                                        <ReactMarkdown
                                            remarkPlugins={[remarkGfm, remarkMath]}
                                            rehypePlugins={[rehypeKatex]}
                                            components={{
                                                a: ({ node, ...props }) => (
                                                    <a {...props} target="_blank" rel="noreferrer" className="text-primary underline hover:opacity-80 break-all" />
                                                ),
                                                code({ node, inline, className, children, ...props }: any) {
                                                    const match = /language-(\w+)/.exec(className || '');
                                                    if (!inline && match) {
                                                        const block = { language: match[1], code: String(children).replace(/\n$/, '') };
                                                        return <CodeBlock block={block} />;
                                                    }
                                                    return <code className="bg-white/10 rounded px-1.5 py-0.5 text-primary/90 text-xs font-mono" {...props}>{children}</code>;
                                                },
                                                table({ children }) {
                                                    return (
                                                        <div className="overflow-x-auto my-4 rounded-xl border border-white/10 bg-white/5">
                                                            <table className="w-full text-left text-sm text-foreground/90">{children}</table>
                                                        </div>
                                                    );
                                                },
                                                thead({ children }) {
                                                    return <thead className="text-xs uppercase bg-black/60 text-primary/90">{children}</thead>;
                                                },
                                                tbody({ children }) {
                                                    return <tbody className="divide-y divide-white/5 text-foreground/90">{children}</tbody>;
                                                },
                                                tr({ children }) {
                                                    return <tr className="hover:bg-white/5 even:bg-white/[0.02]">{children}</tr>;
                                                },
                                                th({ children }) {
                                                    return <th className="px-4 py-3.5 font-semibold whitespace-nowrap">{children}</th>;
                                                },
                                                td({ children }) {
                                                    return <td className="px-4 py-3.5 whitespace-normal">{children}</td>;
                                                },
                                            }}
                                        >
                                            {msg.content || (isStreaming && msg.role === 'assistant' ? '▌' : '')}
                                        </ReactMarkdown>
                                    </div>
                                    {msg.role === 'assistant' && msg.sources && Object.keys(msg.sources).length > 0 && (
                                        <div className="mt-5 pt-4 border-t border-border">
                                            <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground mb-2.5">Sources</p>
                                            <div className="flex flex-wrap gap-2">
                                                {msg.sources.urls && Array.isArray(msg.sources.urls) && msg.sources.urls.map((url: string, i: number) => (
                                                    <a
                                                        key={i}
                                                        href={url}
                                                        target="_blank"
                                                        rel="noreferrer"
                                                        className="text-xs px-2 py-1 rounded-md bg-primary/10 border border-primary/20 hover:bg-primary/20 text-primary truncate max-w-[180px]"
                                                        title={url}
                                                    >
                                                        {(() => {
                                                            try { return new URL(url).hostname; } catch { return url; }
                                                        })()}
                                                    </a>
                                                ))}
                                                {msg.sources.documents && Array.isArray(msg.sources.documents) && msg.sources.documents.map((doc: string, i: number) => (
                                                    <span key={i} className="text-xs px-2 py-1 rounded-md bg-primary/10 border border-primary/20 text-primary/80 truncate max-w-[180px]" title={doc}>
                                                        📄 {String(doc).split('::')[0]}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            <div className="shrink-0 border-t border-border bg-background/95 px-5 pt-4 pb-4">
                {attachedFiles.length > 0 && (
                    <div className="flex flex-wrap gap-2.5 mb-3.5">
                        {attachedFiles.map((f, i) => (
                            <span
                                key={i}
                                className="inline-flex items-center gap-2 text-[13px] px-3 py-2 rounded-lg bg-secondary border border-border"
                            >
                                <FileUp className="w-4 h-4 text-muted-foreground shrink-0" />
                                <span className="truncate max-w-[140px]">{f.name}</span>
                                {onAttachedFilesChange && (
                                    <button
                                        type="button"
                                        onClick={() => removeFile(i)}
                                        className="p-0.5 rounded text-muted-foreground hover:text-foreground hover:bg-black/10"
                                        aria-label="Remove file"
                                    >
                                        <X className="w-3 h-3" />
                                    </button>
                                )}
                            </span>
                        ))}
                    </div>
                )}
                <form onSubmit={onSend} className="flex items-center gap-3 rounded-xl border border-border bg-card p-3">
                    {onAttachedFilesChange && (
                        <>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept={ACCEPT_FILES}
                                multiple
                                className="hidden"
                                onChange={handleFileChange}
                            />
                            <button
                                type="button"
                                onClick={() => fileInputRef.current?.click()}
                                className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors shrink-0"
                                title="Attach files (PDF, DOCX, TXT, HTML)"
                                aria-label="Attach files"
                            >
                                <Paperclip className="w-5 h-5" />
                            </button>
                        </>
                    )}
                    {showResearchToggle && (
                        <div className="flex items-center gap-0.5 rounded-lg bg-secondary/50 p-0.5 shrink-0">
                            <button
                                type="button"
                                onClick={() => {
                                    setIsResearchMode(false);
                                    setIsWebMode(false);
                                }}
                                className={`px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
                                    !isResearchMode && !isWebMode ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
                                }`}
                                aria-pressed={!isResearchMode && !isWebMode}
                                aria-label="Chat only"
                            >
                                Chat
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    setIsResearchMode(false);
                                    setIsWebMode(true);
                                }}
                                className={`px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
                                    isWebMode && !isResearchMode ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
                                }`}
                                aria-pressed={isWebMode && !isResearchMode}
                                aria-label="Web search"
                            >
                                Web
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    setIsResearchMode(true);
                                    setIsWebMode(true);
                                }}
                                className={`px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
                                    isResearchMode ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:text-foreground'
                                }`}
                                aria-pressed={isResearchMode}
                                aria-label="Deep research"
                            >
                                Research
                            </button>
                        </div>
                    )}
                    {optionsTrigger && <div className="shrink-0">{optionsTrigger}</div>}
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder={placeholder}
                        className="flex-1 min-w-0 bg-transparent border-none outline-none focus:ring-0 px-3 py-3 text-foreground text-[15px] placeholder:text-muted-foreground/50 leading-[1.5]"
                        disabled={isStreaming}
                        spellCheck={false}
                        aria-label="Message"
                    />
                    <button
                        type="submit"
                        disabled={(!input.trim() && attachedFiles.length === 0) || isStreaming}
                        className="p-3 rounded-xl shrink-0 text-primary hover:bg-primary/10 disabled:opacity-50 transition-colors"
                        aria-label="Send"
                    >
                        {isStreaming ? <Loader2 className="w-5 h-5 animate-spin" aria-hidden /> : <ArrowUp className="w-5 h-5" aria-hidden />}
                    </button>
                </form>
                <p className="text-[11px] text-muted-foreground text-center mt-3 leading-[1.5]">Wort can make mistakes. Verify important info.</p>
            </div>
        </div>
    );
}
