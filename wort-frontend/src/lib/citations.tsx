import React from 'react';

export function renderWithCitations(text: string, globalSources?: string[]) {
    if (typeof text !== 'string' || !globalSources || globalSources.length === 0) return text;

    const parts = text.split(/(\[[\d\s,]+\](?!\())/g);

    return parts.map((part, index) => {
        const match = part.match(/^\[([\d\s,]+)\]$/);
        if (match) {
            const inner = match[1];
            const nums = inner.split(',').map(n => parseInt(n.trim())).filter(n => !isNaN(n));
            if (nums.length > 0) {
                return (
                    <span key={index}>
                        [
                        {nums.map((num, i) => {
                            const idx = num - 1;
                            let element: React.ReactNode = num;
                            if (idx >= 0 && idx < globalSources.length) {
                                const url = globalSources[idx].split('::')[0];
                                if (url.startsWith('http')) {
                                    element = (
                                        <a key={`ref-${i}`} href={url} target="_blank" rel="noreferrer" className="text-primary hover:text-primary/90 hover:underline cursor-pointer group-hover:text-primary">
                                            {num}
                                        </a>
                                    );
                                }
                            }
                            return (
                                <React.Fragment key={i}>
                                    {i > 0 && ', '}
                                    {element}
                                </React.Fragment>
                            );
                        })}
                        ]
                    </span>
                );
            }
        }
        return part;
    });
}
