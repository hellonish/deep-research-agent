import { fetchApi } from './client';
import type { Message } from '@/lib/types';

export interface ChatSessionItem {
    id: string;
    title?: string;
    created_at?: string;
    updated_at?: string;
}

export interface ResearchItem {
    id: string;
    session_id?: string;
    query?: string;
    created_at?: string;
}

export async function getChats(): Promise<ChatSessionItem[]> {
    const data = await fetchApi('/history/chats');
    return Array.isArray(data) ? data : [];
}

export async function getResearch(): Promise<ResearchItem[]> {
    const data = await fetchApi('/history/research');
    return Array.isArray(data) ? data : [];
}

export async function getChatMessages(chatId: string): Promise<Message[]> {
    const data = await fetchApi(`/history/chats/${chatId}/messages`);
    const list = Array.isArray(data) ? data : [];
    return list
        .sort((a: { created_at?: string }, b: { created_at?: string }) =>
            new Date(a.created_at || 0).getTime() - new Date(b.created_at || 0).getTime()
        )
        .map((m: { id: string; role: string; content?: string; mode?: string; sources?: unknown; created_at?: string }) => ({
            id: m.id,
            role: m.role === 'assistant' ? 'assistant' : 'user',
            content: m.content ?? '',
            mode: m.mode ?? 'chat',
            sources: m.sources && typeof m.sources === 'object' ? m.sources : {},
            created_at: m.created_at ?? new Date().toISOString(),
        }));
}

export async function deleteChat(id: string): Promise<void> {
    await fetchApi(`/history/chats/${id}`, { method: 'DELETE' });
}

export async function deleteResearch(id: string): Promise<void> {
    await fetchApi(`/history/research/${id}`, { method: 'DELETE' });
}

export async function updateChatTitle(id: string, title: string): Promise<void> {
    await fetchApi(`/history/chats/${id}`, {
        method: 'PUT',
        body: JSON.stringify({ title }),
    });
}

export async function updateResearchTitle(id: string, title: string): Promise<void> {
    await fetchApi(`/history/research/${id}`, {
        method: 'PUT',
        body: JSON.stringify({ title }),
    });
}
