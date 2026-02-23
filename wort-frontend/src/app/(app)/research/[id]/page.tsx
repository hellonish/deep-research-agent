"use client";

import { useAuth } from '@/components/AuthProvider';
import { API_BASE, fetchApi } from '@/lib/api';
import { Message } from '@/lib/types';
import { ChatPanel } from '@/components/chat/ChatPanel';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { useParams } from 'next/navigation';
import { useEffect, useState, useRef } from 'react';
import { Panel, Group, Separator } from 'react-resizable-panels';
import { TextBlock } from '@/components/blocks/TextBlock';
import { TableBlock } from '@/components/blocks/TableBlock';
import { ChartBlock } from '@/components/blocks/ChartBlock';
import { CodeBlock } from '@/components/blocks/CodeBlock';
import { SourceListBlock } from '@/components/blocks/SourceListBlock';
import { ResearchEventTree } from '@/components/research/ResearchEventTree';

interface ProgressEvent {
    type: string;
    topic?: string;
    depth?: number;
    message?: string;
    report?: any;
    error?: string;
    data?: any;
    count?: number;
    probe?: string;
    tool?: string;
    query?: string;
    node_id?: string;
    probes?: { node_id: string; probe: string }[];
    completed?: { node_id: string; probe: string }[];
    total_in_level?: number;
}

export default function ResearchJobPage() {
    const params = useParams();
    const { token } = useAuth();
    const jobId = params?.id as string;

    const [status, setStatus] = useState<'connecting' | 'running' | 'complete' | 'failed'>('connecting');
    const [logs, setLogs] = useState<{ id: string, text: string, type: string, depth: number }[]>([]);
    const [report, setReport] = useState<any | null>(null);
    const [errorMsg, setErrorMsg] = useState('');
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [sessionMessages, setSessionMessages] = useState<Message[]>([]);
    const [chatInput, setChatInput] = useState('');
    const [isChatStreaming, setIsChatStreaming] = useState(false);

    const logsEndRef = useRef<HTMLDivElement>(null);
    const chatMessagesEndRef = useRef<HTMLDivElement>(null);
    const ws = useRef<WebSocket | null>(null);

    // Auto-scroll logs
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    useEffect(() => {
        if (!jobId || !token) return;

        const init = async () => {
            try {
                // Fetch initial status first
                const data = await fetchApi(`/chat/research/result/${jobId}`);
                if (data.session_id) {
                    setSessionId(data.session_id);
                    try {
                        const historyData = await fetchApi(`/history/chats/${data.session_id}/messages`);
                        if (Array.isArray(historyData)) {
                            const msgs: Message[] = historyData.map((m: any) => ({
                                id: m.id || `hist_${Math.random()}`,
                                role: m.role === 'user' ? 'user' : 'assistant',
                                content: m.content || '',
                                mode: m.mode || 'chat',
                                sources: m.sources || {},
                                created_at: m.created_at || new Date().toISOString(),
                            }));
                            setSessionMessages(msgs);
                        }
                    } catch (historyErr) {
                        console.error('Failed to hydrate chat history:', historyErr);
                    }
                }

                if (data.status === 'complete') {
                    setStatus('complete');
                    setReport(data.report);
                    return; // Skip WebSocket if already complete
                } else if (data.status === 'failed') {
                    setStatus('failed');
                    setErrorMsg(data.error || 'System Failure');
                    return;
                }

                // If pending/running, connect WebSocket
                connectWebSocket();
            } catch (err) {
                console.error('Failed to fetch initial status:', err);
                connectWebSocket(); // Fallback to WS
            }
        };

        const connectWebSocket = () => {
            // WS URL calculation
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            // Fallback API_BASE parsing for WS
            const apiHost = API_BASE.replace(/^https?:\/\//, '');
            const wsUrl = `${wsProtocol}//${apiHost}/chat/research/stream/${jobId}?token=${token}`;

            ws.current = new WebSocket(wsUrl);

            ws.current.onopen = () => {
                setStatus('running');
                addLog('System', 'Connected to Orchestrator Node.', 0);
            };

            ws.current.onmessage = (event) => {
                try {
                    const data: ProgressEvent = JSON.parse(event.data);

                    if (data.type === 'start') {
                        addLog('Orchestrator', `Initializing traversal network for: ${data.topic}`, 0);
                    } else if (data.type === 'phase_start') {
                        addLog('Phase', data.message || `Starting phase: ${data.data?.phase}`, 0);
                    } else                     if (data.type === 'plan_ready') {
                        addLog('Planner', `Strategic plan formulated. Generated ${data.data?.count || data.count || ''} target vectors.`, 0);
                    } else if (data.type === 'level_start') {
                        const n = data.total_in_level ?? data.probes?.length ?? 0;
                        const d = data.depth ?? 0;
                        addLog('Researcher', `Researching ${n} topic${n !== 1 ? 's' : ''} (depth ${d})…`, d);
                    } else if (data.type === 'level_complete') {
                        const n = data.completed?.length ?? 0;
                        const d = data.depth ?? 0;
                        addLog('Reviewer', `Completed ${n} topic${n !== 1 ? 's' : ''} at depth ${d}.`, d);
                    } else if (data.type === 'probe_start') {
                        addLog('Researcher', `Deploying agent for probe: ${data.probe}`, data.depth || 1);
                    } else if (data.type === 'tool_call') {
                        const prefix = data.node_id != null ? `Node ${data.node_id}: ` : '';
                        addLog('Tool', `${prefix}Executing ${data.tool}: ${data.query}`, data.depth ?? 2);
                    } else if (data.type === 'thinking') {
                        const prefix = data.node_id != null ? `Node ${data.node_id}: ` : '';
                        addLog('LLM', `${prefix}${data.message || 'Analyzing findings for probe...'}`, data.depth ?? 2);
                    } else if (data.type === 'probe_complete') {
                        const prefix = data.node_id != null ? `Node ${data.node_id}: ` : '';
                        addLog('Reviewer', `${prefix}Probe complete. Extracted knowledge.`, data.depth ?? 1);
                    } else if (data.type === 'writing') {
                        addLog('Publisher', `Compiling synthesized intelligence into final report structure.`, 0);
                    } else if (data.type === 'complete') {
                        setStatus('complete');
                        // Always fetch the latest report from the API since WS might not send the full huge JSON
                        fetchReport();
                    } else if (data.type === 'error') {
                        setStatus('failed');
                        setErrorMsg(data.message || 'Unknown error occurred.');
                        addLog('Error', data.message || 'System Failure', 0);
                    } else if (data.message) {
                        // Fallback generic info
                        addLog('Info', data.message, data.depth || 0);
                    }

                } catch (err) {
                    console.error('Failed to parse WS message', err);
                }
            };

            ws.current.onclose = () => {
                // If it closed and we aren't already complete/failed, we might have disconnected
                setStatus((prev) => {
                    if (prev === 'running' || prev === 'connecting') {
                        // Attempt to fetch the final status via API just in case it finished while we disconnected
                        fetchReport();
                        return prev;
                    }
                    return prev;
                });
            };
        };

        init();

        return () => {
            if (ws.current) {
                ws.current.close();
            }
        };
    }, [jobId, token]);


    const fetchReport = async () => {
        try {
            const data = await fetchApi(`/chat/research/result/${jobId}`);
            if (data.session_id) setSessionId(data.session_id);
            if (data.status === 'complete') {
                setStatus('complete');
                setReport(data.report);
            } else if (data.status === 'failed') {
                setStatus('failed');
            }
        } catch (err) {
            console.error('Fetch report error:', err);
        }
    };

    const addLog = (type: string, text: string, depth: number) => {
        setLogs(prev => [...prev, {
            id: Date.now().toString() + Math.random().toString(),
            type,
            text,
            depth
        }]);
    };

    const handleSendChat = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!chatInput.trim() || isChatStreaming || !sessionId) return;
        const userMsg = chatInput.trim();
        setChatInput('');
        setIsChatStreaming(true);
        const tempId = Date.now().toString();
        const streamMsgId = 'stream_' + tempId;
        setSessionMessages((prev) => [
            ...prev,
            { id: tempId, role: 'user', content: userMsg, mode: 'chat', sources: {}, created_at: new Date().toISOString() },
            { id: streamMsgId, role: 'assistant', content: '', mode: 'chat', sources: {}, created_at: new Date().toISOString() },
        ]);
        try {
            const response = await fetch(`${API_BASE}/chat/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                body: JSON.stringify({ message: userMsg, session_id: sessionId, mode: 'chat' }),
            });
            if (!response.ok) throw new Error('Chat API Error');
            if (!response.body) throw new Error('No body');
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
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
                                setSessionMessages((prev) => prev.map((m) => (m.id === streamMsgId ? { ...m, content: finalContent } : m)));
                            } else if (currentEvent === 'sources' && data.urls) {
                                finalSources = { urls: data.urls };
                                setSessionMessages((prev) => prev.map((m) => (m.id === streamMsgId ? { ...m, sources: finalSources } : m)));
                            } else if (currentEvent === 'error' && data.message) {
                                finalContent += `\n\n**Error:** ${data.message}`;
                                setSessionMessages((prev) => prev.map((m) => (m.id === streamMsgId ? { ...m, content: finalContent } : m)));
                            }
                        } catch (_) {}
                    }
                }
            }
        } catch (err) {
            console.error('Follow-up chat error:', err);
            setSessionMessages((prev) => prev.map((m) => (m.id === streamMsgId ? { ...m, content: 'Network error.' } : m)));
        } finally {
            setIsChatStreaming(false);
        }
    };

    return (
        <div className="flex flex-col h-[calc(100vh-2rem)] w-full min-h-0 text-foreground">

            {/* Header */}
            <div className="p-4 border-b border-border/50 bg-card/30 backdrop-blur-md flex items-center justify-between shrink-0">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-primary/10 rounded-xl border border-primary/20 shadow-[0_0_10px_rgba(45,212,191,0.12)]">
                        <Sparkles className="w-6 h-6 text-primary" aria-hidden />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold font-mono tracking-tight">Research Protocol <span className="text-primary drop-shadow-[0_0_6px_rgba(45,212,191,0.4)]">{jobId.split('-')[0]}</span></h1>
                        <p className="text-sm text-muted-foreground flex items-center gap-2">
                            Status:
                            {status === 'connecting' && <span className="text-yellow-500 animate-pulse">Connecting to Matrix...</span>}
                            {status === 'running' && <span className="text-blue-400 animate-pulse flex items-center gap-1"><Loader2 className="w-3 h-3 animate-spin" /> Traversal Active</span>}
                            {status === 'complete' && <span className="text-primary flex items-center gap-1"><CheckCircle2 className="w-3 h-3" /> Synthesis Complete</span>}
                            {status === 'failed' && <span className="text-red-400 flex items-center gap-1"><AlertCircle className="w-3 h-3" /> System Failure</span>}
                        </p>
                    </div>
                </div>
            </div>

            <div className="flex-1 flex w-full min-h-0 px-4 lg:px-6 py-4">
                {(status === 'complete' && report) ? (
                    <Group orientation="horizontal" className="w-full h-full min-h-0 rounded-xl border border-primary/30 overflow-hidden">

                        {/* Left Panel: Same Chat component as main chat */}
                        <Panel defaultSize={40} minSize={20} className="flex flex-col min-h-0 bg-black/95">
                            <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
                                <ChatPanel
                                    messages={sessionMessages}
                                    input={chatInput}
                                    setInput={setChatInput}
                                    onSend={handleSendChat}
                                    isWebMode={false}
                                    setIsWebMode={() => {}}
                                    isResearchMode={false}
                                    setIsResearchMode={() => {}}
                                    isStreaming={isChatStreaming}
                                    placeholder="Follow-up question..."
                                    showResearchToggle={false}
                                    messagesEndRef={chatMessagesEndRef}
                                />
                            </div>
                        </Panel>

                        {/* Interactive Resize Handle */}
                        <Separator className="w-2 bg-secondary hover:bg-primary/30 active:bg-primary/80 transition-colors flex flex-col items-center justify-center cursor-col-resize border-x border-primary/30 group">
                            <div className="w-1 h-8 bg-primary/30 rounded-full group-hover:bg-primary/90 transition-colors" />
                            <div className="w-1 h-8 bg-primary/30 rounded-full mt-1 group-hover:bg-primary/90 transition-colors" />
                        </Separator>

                        {/* Right Panel: The Report */}
                        <Panel defaultSize={60} minSize={30} collapsible={true} className="flex flex-col min-h-0 bg-background">
                            <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden p-6 lg:p-10 scroll-smooth">
                                <motion.div
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ duration: 0.5 }}
                                    className="max-w-4xl mx-auto"
                                >
                                    {/* Report Header */}
                                    <div className="mb-12 pb-8 border-b border-primary/10 text-center">
                                        <div className="inline-flex p-3 bg-primary/10 rounded-2xl border border-primary/20 mb-6 drop-shadow-[0_0_8px_rgba(45,212,191,0.25)]">
                                            <FileText className="w-8 h-8 text-primary" />
                                        </div>
                                        <h2 className="text-4xl font-extrabold tracking-tight mb-4 text-foreground">{report.title}</h2>
                                        <div className="flex items-center justify-center gap-4 text-sm text-primary/80 font-mono">
                                            <span className="bg-primary/10 px-3 py-1 rounded border border-primary/20">Synthesized Intelligence</span>
                                            <span>{new Date().toLocaleDateString()}</span>
                                        </div>
                                        <p className="mt-8 text-lg text-foreground/70 leading-loose text-left max-w-3xl mx-auto border-l-4 border-primary/50 pl-6 py-3 bg-primary/5 rounded-r-lg">
                                            {report.summary}
                                        </p>
                                    </div>

                                    {/* Report Body */}
                                    <div className="space-y-20">
                                        {(() => {
                                            const sourceListBlock = report.blocks.find((b: any) => b.block_type === 'source_list');
                                            const globalSources = sourceListBlock ? sourceListBlock.sources : [];

                                            return report.blocks.map((block: any, i: number) => {
                                                switch (block.block_type) {
                                                    case 'text':
                                                        return <TextBlock key={i} block={block} globalSources={globalSources} />;
                                                    case 'table':
                                                        return <TableBlock key={i} block={block} globalSources={globalSources} />;
                                                    case 'chart':
                                                        return <ChartBlock key={i} block={block} globalSources={globalSources} />;
                                                    case 'code':
                                                        return <CodeBlock key={i} block={block} />;
                                                    case 'source_list':
                                                        return <SourceListBlock key={i} block={block} />;
                                                    default:
                                                        return null;
                                                }
                                            });
                                        })()}
                                    </div>

                                    {/* End of Report */}
                                    <div className="mt-20 pt-10 border-t border-primary/20 text-center flex flex-col items-center">
                                        <div className="w-16 h-1 bg-primary/20 rounded-full mb-8 shadow-[0_0_8px_rgba(45,212,191,0.2)]" />
                                        <p className="font-mono text-xs tracking-widest uppercase text-primary/50">— End of Report —</p>
                                    </div>
                                </motion.div>
                            </div>
                        </Panel>

                    </Group>
                ) : (
                    <div className="flex-1 flex flex-col min-h-0 w-full min-w-0 bg-black/95 rounded-xl border border-primary/30 overflow-hidden">
                        <div className="shrink-0 px-3 py-2 border-b border-primary/20 flex items-center justify-between">
                            <span className="text-xs font-semibold uppercase tracking-wider text-primary/90">Research stream</span>
                            {status === 'running' && (
                                <span className="text-[10px] text-primary/80 flex items-center gap-1">
                                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                                    Live
                                </span>
                            )}
                        </div>
                        <ResearchEventTree logs={logs} isStreaming={status === 'running' || status === 'connecting'} />
                        <div ref={logsEndRef} className="shrink-0" />
                        {status === 'running' && (
                            <div className="shrink-0 px-3 py-2 border-t border-primary/20 flex items-center gap-2 text-xs text-muted-foreground animate-pulse">
                                <span className="inline-block w-2 h-2 rounded-full bg-primary animate-ping" />
                                Awaiting next event…
                            </div>
                        )}
                    </div>
                )}
            </div>

        </div>
    );
}
