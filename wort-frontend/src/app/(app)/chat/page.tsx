"use client";

import { useAuth } from '@/components/AuthProvider';
import { API_BASE, fetchApi } from '@/lib/api';
import { Message } from '@/lib/types';
import { ChatPanel } from '@/components/chat/ChatPanel';
import {
    estimateLLMCalls,
    estimateToolCalls,
    estimateCostUSD,
    DEFAULT_RESEARCH_PARAMS,
    ResearchParams,
} from '@/lib/researchEstimate';
import { ChevronDown, ChevronUp, Loader2, Settings2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';

interface ModelOption {
    id: string;
    name: string;
    description?: string;
}

function StepperInput({
    value,
    min,
    max,
    onChange,
    label,
}: {
    value: number;
    min: number;
    max: number;
    onChange: (v: number) => void;
    label: string;
}) {
    const clamped = (v: number) => Math.min(max, Math.max(min, v));
    return (
        <div>
            <label className="block text-muted-foreground mb-1 text-xs">{label}</label>
            <div className="flex rounded-lg border border-border bg-background overflow-hidden w-fit">
                <input
                    type="number"
                    min={min}
                    max={max}
                    value={value}
                    onChange={(e) => onChange(clamped(parseInt(e.target.value, 10) || min))}
                    className="w-11 py-1.5 px-2 text-center text-foreground bg-transparent border-r border-border text-sm [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                />
                <div className="flex flex-col w-7 shrink-0 border-l border-border">
                    <button
                        type="button"
                        onClick={() => onChange(clamped(value + 1))}
                        disabled={value >= max}
                        className="p-0.5 text-muted-foreground hover:text-foreground hover:bg-white/5 disabled:opacity-40 flex items-center justify-center border-b border-border/50"
                    >
                        <ChevronUp className="w-3.5 h-3.5" />
                    </button>
                    <button
                        type="button"
                        onClick={() => onChange(clamped(value - 1))}
                        disabled={value <= min}
                        className="p-0.5 text-muted-foreground hover:text-foreground hover:bg-white/5 disabled:opacity-40 flex items-center justify-center"
                    >
                        <ChevronDown className="w-3.5 h-3.5" />
                    </button>
                </div>
            </div>
        </div>
    );
}

export default function ChatPage() {
    const params = useParams();
    const router = useRouter();
    const { token, user } = useAuth();
    const chatId = params?.id as string | undefined;

    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isWebMode, setIsWebMode] = useState(false);
    const [isResearchMode, setIsResearchMode] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const [attachedFiles, setAttachedFiles] = useState<File[]>([]);

    const [models, setModels] = useState<ModelOption[]>([]);
    const [selectedModelId, setSelectedModelId] = useState<string | null>(user?.selected_model ?? null);
    const [researchParams, setResearchParams] = useState<ResearchParams>(DEFAULT_RESEARCH_PARAMS);
    const [optionsOpen, setOptionsOpen] = useState(false);
    const optionsRef = useRef<HTMLDivElement>(null);

    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (chatId) loadHistory();
        else setMessages([]);
    }, [chatId]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isStreaming]);

    useEffect(() => {
        if (!token) return;
        fetchApi('/models/available')
            .then((r: { models?: ModelOption[] }) => setModels(r.models || []))
            .catch(() => {});
        if (user?.selected_model) setSelectedModelId(user.selected_model);
    }, [token, user?.selected_model]);

    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (optionsRef.current && !optionsRef.current.contains(e.target as Node)) setOptionsOpen(false);
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const loadHistory = async () => {
        if (!chatId) return;
        try {
            const data = await fetchApi(`/history/chats/${chatId}/messages`);
            const list = Array.isArray(data) ? data : [];
            const sorted = list
                .sort((a: any, b: any) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
                .map((m: any) => ({
                    id: m.id,
                    role: m.role === 'assistant' ? 'assistant' : 'user',
                    content: m.content ?? '',
                    mode: m.mode ?? 'chat',
                    sources: m.sources && typeof m.sources === 'object' ? m.sources : {},
                    created_at: m.created_at ?? new Date().toISOString(),
                }));
            setMessages(sorted);
        } catch (err) {
            console.error('Failed to load history', err);
            setMessages([]);
        }
    };

    const handleSend = async (e: React.FormEvent) => {
        e.preventDefault();
        if (isStreaming) return;

        const hasText = input.trim().length > 0;
        const hasFiles = attachedFiles.length > 0;
        if (!hasText && !hasFiles) return;

        const userMessage = input.trim() || '(Attached files)';
        const filesToUpload = [...attachedFiles];
        setInput('');
        setAttachedFiles([]);

        if (isResearchMode) {
            if (!hasText) return;
            setIsStreaming(true);
            try {
                const data = await fetchApi('/chat/research', {
                    method: 'POST',
                    body: JSON.stringify({
                        query: userMessage,
                        model_id: selectedModelId || undefined,
                        config: {
                            num_plan_steps: researchParams.num_plan_steps,
                            max_depth: researchParams.max_depth,
                            max_probes: researchParams.max_probes,
                            max_tool_pairs: researchParams.max_tool_pairs,
                        },
                    }),
                });
                if (data.job_id) router.push(`/research/${data.job_id}`);
            } catch (err) {
                console.error('Failed to start research', err);
                setIsStreaming(false);
            }
            return;
        }

        if (filesToUpload.length > 0) {
            for (const file of filesToUpload) {
                const formData = new FormData();
                formData.append('file', file);
                try {
                    await fetchApi('/ingest/upload', { method: 'POST', body: formData });
                } catch (err) {
                    console.error('Ingest failed for', file.name, err);
                }
            }
        }

        setMessages((prev) => [
            ...prev,
            { id: Date.now().toString(), role: 'user', content: userMessage, mode: isWebMode ? 'web' : 'standard', sources: {}, created_at: new Date().toISOString() },
        ]);
        setIsStreaming(true);
        const streamMsgId = 'stream_' + Date.now();
        setMessages((prev) => [
            ...prev,
            { id: streamMsgId, role: 'assistant', content: '', mode: isWebMode ? 'web' : 'standard', sources: {}, created_at: new Date().toISOString() },
        ]);

        try {
            const response = await fetch(`${API_BASE}/chat/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
                body: JSON.stringify({
                    message: userMessage,
                    session_id: chatId || null,
                    mode: isWebMode ? 'web' : 'chat',
                    model_id: selectedModelId || undefined,
                }),
            });
            if (!response.ok) throw new Error(`Chat API error: ${response.statusText}`);
            if (!response.body) throw new Error('No response body');

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let sessionId = chatId;
            let finalContent = '';
            let finalSources: Record<string, unknown> = {};

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                let currentEvent = 'message';
                for (const line of chunk.split('\n')) {
                    if (line.startsWith('event: ')) currentEvent = line.replace('event: ', '').trim();
                    else if (line.startsWith('data: ')) {
                        const dataStr = line.replace('data: ', '').trim();
                        if (!dataStr) continue;
                        try {
                            const data = JSON.parse(dataStr);
                            if (currentEvent === 'token' && data.content) {
                                finalContent += data.content;
                                setMessages((prev) => prev.map((m) => (m.id === streamMsgId ? { ...m, content: finalContent } : m)));
                            } else if (currentEvent === 'sources' && data.urls) {
                                finalSources = { urls: data.urls };
                                setMessages((prev) => prev.map((m) => (m.id === streamMsgId ? { ...m, sources: finalSources } : m)));
                            } else if (currentEvent === 'done' && data.session_id) sessionId = data.session_id;
                            else if (currentEvent === 'error' && data.message) {
                                finalContent += `\n\n**Error:** ${data.message}`;
                                setMessages((prev) => prev.map((m) => (m.id === streamMsgId ? { ...m, content: finalContent } : m)));
                            }
                        } catch (_) {}
                    }
                }
            }
            if (!chatId && sessionId) router.replace(`/chat/${sessionId}`);
        } catch (err) {
            console.error('Chat error', err);
            setMessages((prev) => [
                ...prev,
                { id: 'err_' + Date.now(), role: 'assistant', content: 'Sorry, I encountered an error.', mode: 'standard', sources: {}, created_at: new Date().toISOString() },
            ]);
        } finally {
            setIsStreaming(false);
        }
    };

    const llmEst = estimateLLMCalls(researchParams);
    const toolEst = estimateToolCalls(researchParams);
    const costEst = estimateCostUSD(researchParams, llmEst);

    const placeholder = isResearchMode
        ? 'Describe what to research…'
        : isWebMode
          ? 'Ask with web search…'
          : 'Message Wort…';

    const optionsTrigger = (
        <div className="relative" ref={optionsRef}>
            <button
                type="button"
                onClick={() => setOptionsOpen((o) => !o)}
                className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors shrink-0"
                aria-label="Options"
                aria-expanded={optionsOpen}
            >
                <Settings2 className="w-5 h-5" />
            </button>
            {optionsOpen && (
                <div className="absolute right-0 bottom-full mb-1 w-72 rounded-xl border border-border bg-card shadow-xl p-3 z-50">
                    <p className="text-xs font-medium text-muted-foreground mb-2">Model</p>
                    <select
                        value={selectedModelId || ''}
                        onChange={(e) => setSelectedModelId(e.target.value || null)}
                        className="w-full rounded-lg border border-border bg-background px-3 py-2 text-foreground text-sm mb-3"
                    >
                        <option value="">Default</option>
                        {models.map((m) => (
                            <option key={m.id} value={m.id}>{m.name || m.id}</option>
                        ))}
                    </select>
                    {isResearchMode && (
                        <>
                            <p className="text-xs font-medium text-muted-foreground mb-2 mt-2">Research params</p>
                            <div className="grid grid-cols-2 gap-2 mb-2">
                                <StepperInput label="Plan steps" value={researchParams.num_plan_steps} min={1} max={15} onChange={(v) => setResearchParams((p) => ({ ...p, num_plan_steps: v }))} />
                                <StepperInput label="Max depth" value={researchParams.max_depth} min={1} max={10} onChange={(v) => setResearchParams((p) => ({ ...p, max_depth: v }))} />
                                <StepperInput label="Max probes" value={researchParams.max_probes} min={1} max={10} onChange={(v) => setResearchParams((p) => ({ ...p, max_probes: v }))} />
                                <StepperInput label="Tool pairs" value={researchParams.max_tool_pairs} min={1} max={10} onChange={(v) => setResearchParams((p) => ({ ...p, max_tool_pairs: v }))} />
                            </div>
                            <p className="text-[10px] text-muted-foreground">
                                Est. ~{llmEst} LLM · {toolEst} tools · ~${costEst.toFixed(2)}
                            </p>
                        </>
                    )}
                </div>
            )}
        </div>
    );

    return (
        <div className="flex flex-col h-full w-full min-h-0 px-4 lg:px-6 py-4">
            <div className="flex-1 min-h-0 flex flex-col mx-auto w-full max-w-5xl rounded-2xl border border-border bg-card/50 overflow-hidden shadow-sm">
                <ChatPanel
                    messages={messages}
                    input={input}
                    setInput={setInput}
                    onSend={handleSend}
                    isWebMode={isWebMode}
                    setIsWebMode={setIsWebMode}
                    isResearchMode={isResearchMode}
                    setIsResearchMode={setIsResearchMode}
                    isStreaming={isStreaming}
                    placeholder={placeholder}
                    showResearchToggle={true}
                    messagesEndRef={messagesEndRef}
                    attachedFiles={attachedFiles}
                    onAttachedFilesChange={setAttachedFiles}
                    optionsTrigger={optionsTrigger}
                />
            </div>
        </div>
    );
}
