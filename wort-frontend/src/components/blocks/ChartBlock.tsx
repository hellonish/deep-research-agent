import React, { useState, useEffect } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    LineChart, Line, PieChart, Pie, Cell
} from 'recharts';
import { renderWithCitations } from '@/lib/citations';

export function ChartBlock({ block, globalSources }: { block: any, globalSources?: string[] }) {
    const [isMounted, setIsMounted] = useState(false);

    // recharts has hydration issues with ResponsiveContainer sometimes, so render only after mount
    useEffect(() => {
        setIsMounted(true);
    }, []);

    if (!isMounted) return <div className="h-[300px] w-full animate-pulse bg-primary/10 rounded-xl mt-8"></div>;

    if (!block.datasets || block.datasets.length === 0 || !block.labels) return null;

    // Transform datasets to Recharts expected format
    const data = block.labels.map((label: string, i: number) => {
        const dataPoint: any = { name: label };
        block.datasets.forEach((ds: any) => {
            dataPoint[ds.label] = ds.data[i];
        });
        return dataPoint;
    });

    const colors = ['#22d3ee', '#818cf8', '#34d399', '#f472b6', '#fbbf24', '#f87171'];

    const renderChart = () => {
        const type = block.chart_type?.toLowerCase() || 'bar';

        switch (type) {
            case 'line':
                return (
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff1a" vertical={false} />
                            <XAxis dataKey="name" stroke="#67e8f9" tick={{ fill: '#67e8f9', fontSize: 12 }} tickMargin={10} />
                            <YAxis stroke="#67e8f9" tick={{ fill: '#67e8f9', fontSize: 12 }} tickMargin={10} axisLine={false} tickLine={false} />
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#22d3ee', color: '#cffafe', borderRadius: '8px' }} itemStyle={{ color: '#cffafe' }} labelFormatter={(label) => renderWithCitations(label, globalSources)} formatter={(value, name) => [value, renderWithCitations(name as string, globalSources)]} />
                            <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="circle" />
                            {block.datasets.map((ds: any, idx: number) => (
                                <Line key={ds.label} type="monotone" dataKey={ds.label} stroke={colors[idx % colors.length]} strokeWidth={3} dot={{ r: 4, strokeWidth: 2 }} activeDot={{ r: 6 }} />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                );
            case 'pie':
                const pieData = block.labels.map((label: string, i: number) => ({
                    name: label,
                    value: block.datasets[0].data[i]
                }));
                return (
                    <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#22d3ee', borderRadius: '8px' }} itemStyle={{ color: '#cffafe' }} formatter={(value, name) => [value, renderWithCitations(name as string, globalSources)]} />
                            <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="circle" />
                            <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} innerRadius={60} label={false}>
                                {pieData.map((entry: any, index: number) => (
                                    <Cell key={`cell-${index}`} fill={colors[index % colors.length]} stroke="#000" strokeWidth={2} />
                                ))}
                            </Pie>
                        </PieChart>
                    </ResponsiveContainer>
                );
            case 'bar':
            default:
                return (
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff1a" vertical={false} />
                            <XAxis dataKey="name" stroke="#67e8f9" tick={{ fill: '#67e8f9', fontSize: 12 }} tickMargin={10} />
                            <YAxis stroke="#67e8f9" tick={{ fill: '#67e8f9', fontSize: 12 }} tickMargin={10} axisLine={false} tickLine={false} />
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#22d3ee', color: '#cffafe', borderRadius: '8px' }} itemStyle={{ color: '#cffafe' }} cursor={{ fill: '#ffffff0a' }} labelFormatter={(label) => renderWithCitations(label, globalSources)} formatter={(value, name) => [value, renderWithCitations(name as string, globalSources)]} />
                            <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="circle" />
                            {block.datasets.map((ds: any, idx: number) => (
                                <Bar key={ds.label} dataKey={ds.label} fill={colors[idx % colors.length]} radius={[4, 4, 0, 0]} />
                            ))}
                        </BarChart>
                    </ResponsiveContainer>
                );
        }
    };

    return (
        <section className="mt-8 rounded-xl border border-white/10 bg-black/40 p-6 shadow-[0_0_15px_rgba(34,211,238,0.05)] w-full">
            {block.title && (
                <h3 className="text-xl font-bold mb-6 text-foreground tracking-tight mt-0">{renderWithCitations(block.title.replace(/^#+\s*/, ''), globalSources)}</h3>
            )}
            <div className="w-full text-foreground/90 font-sans">
                {renderChart()}
            </div>
        </section>
    );
}
