import { API_BASE, fetchApi } from './client';

export interface StartResearchBody {
    query: string;
    model_id?: string;
    config?: {
        num_plan_steps?: number;
        max_depth?: number;
        max_probes?: number;
        max_tool_pairs?: number;
    };
}

export interface StartResearchResponse {
    job_id: string;
    session_id?: string;
}

export interface ResearchResultResponse {
    status: 'pending' | 'running' | 'complete' | 'failed';
    session_id?: string;
    report?: unknown;
    error?: string;
}

export interface StreamChatBody {
    message: string;
    session_id: string | null;
    mode?: 'chat' | 'web';
    model_id?: string;
}

/**
 * Returns the raw Response for streaming; caller must read response.body (e.g. getReader()).
 */
export function streamChat(body: StreamChatBody, token: string | null): Promise<Response> {
    const headers: Record<string, string> = {
        'Content-Type': 'application/json',
    };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    return fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
    });
}

/**
 * Start research: POST returns SSE stream; we read the first event ("started") and return job_id + session_id.
 */
export async function startResearch(body: StartResearchBody, token: string | null): Promise<StartResearchResponse> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch(`${API_BASE}/chat/research`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || res.statusText);
    }
    if (!res.body) throw new Error('No response body');

    const data = await readFirstSseEvent(res.body);
    const payload = data as { type?: string; job_id?: string; session_id?: string };
    if (payload.type !== 'started' || !payload.job_id) {
        throw new Error('Invalid stream response');
    }
    return { job_id: payload.job_id, session_id: payload.session_id };
}

/**
 * Read the first SSE "data:" line from a ReadableStream and parse as JSON.
 */
async function readFirstSseEvent(body: ReadableStream<Uint8Array>): Promise<unknown> {
    const reader = body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    try {
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const match = buffer.match(/^data:\s*(.+?)(?:\n|$)/m);
            if (match) {
                const raw = match[1].trim();
                if (raw === '[DONE]' || raw === '') continue;
                return JSON.parse(raw) as unknown;
            }
        }
    } finally {
        reader.releaseLock();
    }
    throw new Error('No SSE data event');
}

export async function getResearchResult(jobId: string): Promise<ResearchResultResponse> {
    return fetchApi(`/chat/research/result/${jobId}`) as Promise<ResearchResultResponse>;
}
