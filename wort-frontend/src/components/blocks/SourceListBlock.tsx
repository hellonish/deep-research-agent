import React from 'react';

export function SourceListBlock({ block }: { block: any }) {
    if (!block.sources || block.sources.length === 0) return null;

    return (
        <section className="pt-8 mt-12 border-t border-primary/20">
            <h3 className="text-xl font-bold mb-4 font-mono uppercase tracking-widest text-primary/80">
                Sources Analyzed
            </h3>
            <ul className="space-y-2">
                {block.sources.map((src: string, j: number) => (
                    <li key={j} className="flex items-start gap-2 text-sm text-foreground/90 group">
                        <div className="w-1.5 h-1.5 rounded-full bg-primary/50 mt-1.5 shrink-0 shadow-[0_0_6px_rgba(45,212,191,0.35)] group-hover:bg-primary transition-colors" />
                        {src.startsWith('http') ? (
                            <a href={src} target="_blank" rel="noreferrer" className="hover:text-primary hover:underline break-words relative z-10 transition-colors">
                                {src}
                            </a>
                        ) : (
                            <span className="break-words">{src.split('::')[0]}</span>
                        )}
                    </li>
                ))}
            </ul>
        </section>
    );
}
