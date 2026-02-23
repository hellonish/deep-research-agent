"use client";

import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

export function CodeBlock({ block }: { block: any }) {
    const [copied, setCopied] = useState(false);

    if (!block.code) return null;

    const handleCopy = () => {
        navigator.clipboard.writeText(block.code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const lang = (block.language || 'text').toLowerCase();

    return (
        <section className="mt-8 rounded-xl border border-primary/20 bg-[#0d1117] overflow-hidden shadow-lg group">
            <div className="flex items-center justify-between px-4 py-2 bg-[#161b22] border-b border-primary/20">
                <span className="text-xs font-mono font-semibold text-primary/80 uppercase tracking-widest">
                    {block.language || 'text'}
                </span>
                <button
                    onClick={handleCopy}
                    className="flex items-center gap-1.5 text-xs text-primary/60 hover:text-primary transition-colors"
                >
                    {copied ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
                    {copied ? 'Copied' : 'Copy'}
                </button>
            </div>
            <div className="p-0 overflow-x-auto relative text-sm">
                <SyntaxHighlighter
                    language={lang}
                    style={oneDark}
                    customStyle={{ margin: 0, padding: '1rem 1.25rem', background: 'transparent', fontSize: '0.875rem', lineHeight: 1.6 }}
                    codeTagProps={{ style: { fontFamily: 'ui-monospace, monospace' } }}
                    showLineNumbers={false}
                    PreTag="div"
                >
                    {block.code}
                </SyntaxHighlighter>
            </div>
            {block.title && (
                <div className="px-4 py-2 border-t border-primary/10 bg-black/50">
                    <p className="text-xs text-muted-foreground italic">{block.title.replace(/^#+\s*/, '')}</p>
                </div>
            )}
        </section>
    );
}
