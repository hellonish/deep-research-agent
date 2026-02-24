import { API_BASE, fetchApi } from './client';

export interface PlanStepDto {
    step_number: number;
    action: string;
    description: string;
}

export interface StartResearchBody {
    query: string;
    model_id?: string;
    config?: {
        num_plan_steps?: number;
        max_depth?: number;
        max_probes?: number;
        max_tool_pairs?: number;
    };
    /** From scoping: use this plan instead of generating a new one. */
    refined_plan?: PlanStepDto[];
    /** User answers to clarifying questions / refinements; scopes the research. */
    user_context?: string;
}

export interface ResearchScopeRequest {
    query: string;
    model_id?: string;
    num_plan_steps?: number;
}

export interface ResearchScopeResponse {
    query_type: string;
    plan: PlanStepDto[];
    clarifying_questions: string[];
}

/**
 * Get a research plan and clarifying questions only (no job).
 * User can refine and then start research with refined_plan + user_context.
 */
export async function researchScope(
    body: ResearchScopeRequest,
    token: string | null
): Promise<ResearchScopeResponse> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}/chat/research/scope`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || res.statusText);
    }
    return res.json() as Promise<ResearchScopeResponse>;
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

/** Progress event payload from the research stream (type + backend-specific fields). */
export type ResearchProgressEvent = { type: string; [key: string]: unknown };

/**
 * Subscribe to the SSE progress stream for a research job. Calls onEvent for each event.
 * Resolves when the stream ends (complete/error) or rejects on fetch/parse errors.
 * Pass signal to abort (e.g. on unmount).
 */
export async function streamResearchProgress(
    jobId: string,
    token: string | null,
    onEvent: (event: ResearchProgressEvent) => void,
    signal?: AbortSignal
): Promise<void> {
    const headers: Record<string, string> = { Accept: 'text/event-stream' };
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch(`${API_BASE}/chat/research/stream/${jobId}`, {
        method: 'GET',
        headers,
        signal,
    });
    if (!res.ok) throw new Error(res.statusText || 'Stream failed');
    if (!res.body) throw new Error('No response body');

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    try {
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const raw = line.slice(6).trim();
                if (raw === '' || raw === '[DONE]') continue;
                try {
                    const event = JSON.parse(raw) as ResearchProgressEvent;
                    onEvent(event);
                    if (event.type === 'complete' || event.type === 'error') return;
                } catch {
                    // ignore malformed lines
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
}
