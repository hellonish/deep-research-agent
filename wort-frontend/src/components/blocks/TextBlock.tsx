import React from "react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { CodeBlock } from './CodeBlock';

interface TextBlockProps {
    block: any;
    globalSources: string[];
}

export function TextBlock({ block, globalSources }: TextBlockProps) {
    let markdownContent = block.markdown || '';
    if (globalSources.length > 0) {
        markdownContent = markdownContent.replace(/\[([\d\s,]+)\](?!\()/g, (match: string, inner: string) => {
            const nums = inner.split(',').map(n => parseInt(n.trim())).filter(n => !isNaN(n));
            if (nums.length === 0) return match;

            const converted = nums.map(num => {
                const idx = num - 1;
                if (idx >= 0 && idx < globalSources.length) {
                    const url = globalSources[idx].split('::')[0];
                    if (url.startsWith('http')) {
                        return `[${num}](${url})`;
                    }
                }
                return `${num}`;
            });
            return `[${converted.join(', ')}]`;
        });
    }

    return (
        <section className="prose prose-lg dark:prose-invert max-w-none prose-headings:font-bold prose-headings:text-foreground prose-headings:tracking-tight prose-a:text-primary hover:prose-a:text-primary/90 prose-p:leading-[1.9] prose-p:my-4 prose-li:leading-[1.85] space-y-8 prose-p:text-foreground/90 prose-li:text-foreground/90 prose-pre:bg-white/5 prose-pre:border prose-pre:border-white/10 rounded-xl mt-8 pt-4">
            {block.title && (() => {
                const match = block.title.match(/^(#{1,6})\s*(.*)/);
                const level = match ? match[1].length : 3;
                const cleanTitle = match ? match[2] : block.title;

                const HeadingTag = `h${level}` as any;
                const sizeClass = {
                    1: 'text-4xl mb-8',
                    2: 'text-3xl mb-6',
                    3: 'text-2xl mb-4',
                    4: 'text-xl mb-3',
                    5: 'text-lg mb-2',
                    6: 'text-base mb-2',
                }[level as 1 | 2 | 3 | 4 | 5 | 6] || 'text-2xl mb-4';

                return (
                    <HeadingTag className={`${sizeClass} font-bold text-foreground tracking-tight mt-0`}>
                        {cleanTitle}
                    </HeadingTag>
                );
            })()}

            <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex]}
                components={{
                    a: ({ node, ...props }) => (
                        <a {...props} target="_blank" rel="noreferrer" className="text-primary hover:text-primary/90 hover:underline break-words drop-shadow-[0_0_6px_rgba(45,212,191,0.25)] relative z-10 font-bold" />
                    ),
                    code({ node, inline, className, children, ...props }: any) {
                        const match = /language-(\w+)/.exec(className || '');
                        if (!inline && match) {
                            const block = {
                                language: match[1],
                                code: String(children).replace(/\n$/, '')
                            };
                            return <CodeBlock block={block} />;
                        }
                        return <code className="bg-white/10 rounded px-1.5 py-0.5 text-foreground text-xs font-mono" {...props}>{children}</code>;
                    },
                    table({ children }) {
                        return (
                            <div className="overflow-x-auto mt-6 mb-6 rounded-xl border border-white/10 bg-white/5 shadow-lg">
                                <table className="w-full text-left text-sm text-foreground/90">
                                    {children}
                                </table>
                            </div>
                        );
                    },
                    thead({ children }) {
                        return <thead className="text-xs uppercase bg-black/60 text-primary/80 sticky top-0">{children}</thead>;
                    },
                    tbody({ children }) {
                        return <tbody className="divide-y divide-white/5 text-foreground/95">{children}</tbody>;
                    },
                    tr({ children }) {
                        return <tr className="hover:bg-white/5 transition-colors even:bg-white/[0.02]">{children}</tr>;
                    },
                    th({ children }) {
                        return <th className="px-6 py-4 font-semibold whitespace-nowrap">{children}</th>;
                    },
                    td({ children }) {
                        return <td className="px-6 py-4 whitespace-normal">{children}</td>;
                    }
                }}
            >
                {markdownContent}
            </ReactMarkdown>
        </section>
    );
}
