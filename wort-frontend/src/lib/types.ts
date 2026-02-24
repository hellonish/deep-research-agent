// Models matching backend responses

export interface User {
    id: string;
    email: string;
    name: string;
    picture: string;
    selected_model: string | null;
}

export interface ChatSession {
    id: string;
    title: string;
    created_at: string;
    updated_at: string;
}

export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    mode: string;
    sources: Record<string, any>;
    created_at: string;
}

export interface ResearchJob {
    id: string;
    query: string;
    status: 'pending' | 'running' | 'complete' | 'failed';
    model_id: string;
    created_at: string;
    completed_at: string | null;
}

export interface AuthResponse {
    access_token: string;
    user: User;
}

export interface ModelOption {
    id: string;
    name: string;
    description?: string;
    provider?: string;
    display_name?: string;
}
