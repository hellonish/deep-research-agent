import React from 'react';
import { renderWithCitations } from '@/lib/citations';

export function TableBlock({ block, globalSources }: { block: any, globalSources?: string[] }) {
    if (!block.headers || !block.rows) return null;

    return (
        <section className="mt-8 pt-4 overflow-hidden rounded-xl border border-white/10 bg-white/5 shadow-lg relative">
            {block.title && (
                <div className="px-6 py-4 border-b border-white/10 bg-black/40">
                    <h3 className="text-xl font-bold text-foreground tracking-tight m-0">{renderWithCitations(block.title.replace(/^#+\s*/, ''), globalSources)}</h3>
                </div>
            )}
            <div className="overflow-x-auto">
                <table className="w-full text-left text-sm text-foreground/90">
                    <thead className="text-xs uppercase bg-black/60 text-primary/80 sticky top-0">
                        <tr>
                            {block.headers.map((h: string, i: number) => (
                                <th key={i} className="px-6 py-4 font-semibold whitespace-nowrap">{renderWithCitations(h, globalSources)}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5 text-foreground/95">
                        {block.rows.map((row: string[], i: number) => (
                            <tr key={i} className="hover:bg-white/5 transition-colors even:bg-white/[0.02]">
                                {row.map((cell: string, j: number) => (
                                    <td key={j} className="px-6 py-4 whitespace-normal">{renderWithCitations(cell, globalSources)}</td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </section>
    );
}
