"use client";

import { useAuth } from '@/components/AuthProvider';
import { getResearchResult, getChatMessages, streamChat, streamResearchProgress, type ResearchProgressEvent } from '@/apis';
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
    const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const streamAbortRef = useRef<AbortController | null>(null);

    const fetchReport = async () => {
        try {
            const data = await getResearchResult(jobId);
            if (data.session_id) setSessionId(data.session_id);
            if (data.status === 'complete') {
                setStatus('complete');
                setReport(data.report);
            } else if (data.status === 'failed') {
                setStatus('failed');
                setErrorMsg(data.error || 'System Failure');
            }
        } catch (err) {
            console.error('Fetch report error:', err);
            const is404 = err instanceof Error && (err.message === 'Research job not found' || err.message.includes('not found'));
            if (is404) {
                setStatus('failed');
                setErrorMsg('Job not found. It may have been created in another session or the link is invalid.');
            }
        }
    };

    // Load initial state and poll when pending/running
    useEffect(() => {
        if (!jobId || !token) return;

        const init = async () => {
            try {
                const data = await getResearchResult(jobId);
                if (data.session_id) {
                    setSessionId(data.session_id);
                    try {
                        const msgs = await getChatMessages(data.session_id);
                        setSessionMessages(msgs);
                    } catch (historyErr) {
                        console.error('Failed to hydrate chat history:', historyErr);
                    }
                }

                if (data.status === 'complete') {
                    setStatus('complete');
                    setReport(data.report);
                    return;
                }
                if (data.status === 'failed') {
                    setStatus('failed');
                    setErrorMsg(data.error || 'System Failure');
                    return;
                }

                setStatus('running');
                addLog('Phase', 'Research in progress…', 0);
                pollRef.current = setInterval(fetchReport, 2500);

                // Subscribe to live progress stream
                const abort = new AbortController();
                streamAbortRef.current = abort;
                streamResearchProgress(jobId, token, (event: ResearchProgressEvent) => {
                    const depth = typeof event.depth === 'number' ? event.depth : 0;
                    if (event.type === 'phase_start') {
                        addLog('Phase', (event.message as string) || event.phase as string, 0);
                    } else if (event.type === 'plan_ready') {
                        const count = event.count as number;
                        addLog('Planner', `Plan ready: ${count} probes`, 0);
                    } else if (event.type === 'level_start') {
                        const d = event.depth as number;
                        const total = event.total_in_level as number;
                        addLog('Orchestrator', `Depth ${d}: ${total} probes`, d);
                    } else if (event.type === 'probe_start') {
                        const probe = (event.probe as string)?.slice(0, 80) || 'Probe';
                        addLog('Researcher', probe + (probe.length >= 80 ? '…' : ''), depth);
                    } else if (event.type === 'tool_call') {
                        const tool = event.tool as string;
                        const query = (event.query as string)?.slice(0, 60) || '';
                        addLog('Tool', `${tool}: ${query}`, depth + 1);
                    } else if (event.type === 'thinking') {
                        addLog('LLM', (event.message as string) || 'Synthesizing…', depth);
                    } else if (event.type === 'probe_complete') {
                        addLog('Researcher', 'Probe complete', depth);
                    } else if (event.type === 'level_complete') {
                        addLog('Orchestrator', 'Level complete', depth);
                    } else if (event.type === 'writing') {
                        addLog('Publisher', (event.message as string) || 'Composing report…', 0);
                    } else if (event.type === 'complete') {
                        addLog('Phase', 'Synthesis complete', 0);
                        setStatus('complete');
                        fetchReport();
                    } else if (event.type === 'error') {
                        addLog('Error', (event.message as string) || 'Error', 0);
                        setStatus('failed');
                        setErrorMsg((event.message as string) || 'System failure');
                    }
                }, abort.signal).catch((err: unknown) => {
                    if (err instanceof Error && err.name !== 'AbortError') console.error('Research stream error:', err);
                }).finally(() => {
                    streamAbortRef.current = null;
                });
            } catch (err) {
                console.error('Failed to fetch initial status:', err);
                const is404 = err instanceof Error && (err.message === 'Research job not found' || err.message.includes('not found'));
                setStatus('failed');
                setErrorMsg(is404 ? 'Job not found. It may have been created in another session or the link is invalid.' : 'Failed to load job');
            }
        };

        init();

        return () => {
            if (pollRef.current) {
                clearInterval(pollRef.current);
                pollRef.current = null;
            }
            if (streamAbortRef.current) {
                streamAbortRef.current.abort();
                streamAbortRef.current = null;
            }
        };
    }, [jobId, token]);

    // Stop polling when status becomes complete/failed (fetchReport updates status)
    useEffect(() => {
        if ((status === 'complete' || status === 'failed') && pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
        }
    }, [status]);

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
            const response = await streamChat(
                { message: userMsg, session_id: sessionId, mode: 'chat' },
                token
            );
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
