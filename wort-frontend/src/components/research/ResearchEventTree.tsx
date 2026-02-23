"use client";

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, ChevronDown, Zap, Brain, Wrench, MessageSquare, FileText, AlertCircle, Info } from 'lucide-react';

export interface LogEntry {
    id: string;
    type: string;
    text: string;
    depth: number;
}

interface TreeNode {
    id: string;
    type: string;
    text: string;
    depth: number;
    children: TreeNode[];
}

function buildTree(logs: LogEntry[]): TreeNode[] {
    if (logs.length === 0) return [];
    const nodes: TreeNode[] = logs.map((l) => ({ ...l, children: [] }));
    const stack: number[] = [];
    const roots: TreeNode[] = [];

    for (let i = 0; i < nodes.length; i++) {
        const depth = nodes[i].depth;
        while (stack.length > 0 && nodes[stack[stack.length - 1]].depth >= depth) stack.pop();
        if (stack.length > 0) {
            const parentIdx = stack[stack.length - 1];
            nodes[parentIdx].children.push(nodes[i]);
        } else {
            roots.push(nodes[i]);
        }
        stack.push(i);
    }
    return roots;
}

function iconForType(type: string) {
    switch (type) {
        case 'Orchestrator':
        case 'Phase':
            return <Zap className="w-3.5 h-3.5 text-amber-400 shrink-0" />;
        case 'Planner':
        case 'Researcher':
        case 'Reviewer':
            return <Brain className="w-3.5 h-3.5 text-blue-400 shrink-0" />;
        case 'Tool':
        case 'LLM':
            return <Wrench className="w-3.5 h-3.5 text-primary shrink-0" />;
        case 'Publisher':
            return <FileText className="w-3.5 h-3.5 text-green-400 shrink-0" />;
        case 'Error':
            return <AlertCircle className="w-3.5 h-3.5 text-red-400 shrink-0" />;
        case 'USER':
        case 'WORT':
            return <MessageSquare className="w-3.5 h-3.5 text-primary shrink-0" />;
        case 'System':
            return <Zap className="w-3.5 h-3.5 text-amber-400 shrink-0" />;
        default:
            return <Info className="w-3.5 h-3.5 text-muted-foreground shrink-0" />;
    }
}

function typeColor(type: string): string {
    switch (type) {
        case 'Error': return 'text-red-400';
        case 'Orchestrator':
        case 'Phase': return 'text-amber-400';
        case 'Planner':
        case 'Researcher':
        case 'Reviewer': return 'text-blue-300';
        case 'Tool':
        case 'LLM': return 'text-primary/90';
        case 'Publisher': return 'text-green-400';
        case 'USER': return 'text-blue-100';
        case 'WORT': return 'text-foreground/95';
        case 'System': return 'text-amber-400/90';
        default: return 'text-primary/90';
    }
}

function TreeRow({ node, level, isLast, isExpanded, onToggle }: {
    node: TreeNode;
    level: number;
    isLast: boolean;
    isExpanded: boolean;
    onToggle: () => void;
}) {
    const hasChildren = node.children.length > 0;
    const [open, setOpen] = useState(true);

    return (
        <div className="select-none">
            <div
                className="flex items-start gap-2 py-1.5 pr-2 group cursor-pointer hover:bg-white/5 rounded"
                style={{ paddingLeft: `${level * 1.25 + 0.5}rem` }}
                onClick={() => hasChildren && setOpen((o) => !o)}
            >
                <span className="flex items-center gap-1 shrink-0 mt-0.5">
                    {hasChildren ? (
                        <button type="button" onClick={(e) => { e.stopPropagation(); setOpen((o) => !o); }} className="p-0.5 rounded hover:bg-white/10">
                            {open ? <ChevronDown className="w-3.5 h-3.5 text-primary/70" /> : <ChevronRight className="w-3.5 h-3.5 text-primary/70" />}
                        </button>
                    ) : (
                        <span className="w-4 inline-block" />
                    )}
                    {iconForType(node.type)}
                </span>
                <div className="flex-1 min-w-0">
                    <span className={`text-[10px] font-semibold uppercase tracking-wider ${typeColor(node.type)}`}>
                        [{node.type}]
                    </span>
                    <span className={`text-xs ml-1.5 ${typeColor(node.type)} opacity-95`}>{node.text}</span>
                </div>
            </div>
            <AnimatePresence>
                {hasChildren && open && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.15 }}
                        className="overflow-hidden"
                    >
                        {node.children.map((child, i) => (
                            <TreeRow
                                key={child.id}
                                node={child}
                                level={level + 1}
                                isLast={i === node.children.length - 1}
                                isExpanded={isExpanded}
                                onToggle={onToggle}
                            />
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

export function ResearchEventTree({ logs, isStreaming }: { logs: LogEntry[]; isStreaming?: boolean }) {
    const roots = buildTree(logs);

    return (
        <div className="font-mono text-sm flex-1 flex flex-col min-h-0 overflow-hidden">
            <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden p-3 space-y-0 scroll-smooth">
                {roots.length === 0 ? (
                    <div className="flex items-center justify-center py-8 text-muted-foreground text-xs">
                        {isStreaming ? 'Waiting for events…' : 'No events yet.'}
                    </div>
                ) : (
                    roots.map((node, i) => (
                        <TreeRow
                            key={node.id}
                            node={node}
                            level={0}
                            isLast={i === roots.length - 1}
                            isExpanded={true}
                            onToggle={() => {}}
                        />
                    ))
                )}
            </div>
        </div>
    );
}
