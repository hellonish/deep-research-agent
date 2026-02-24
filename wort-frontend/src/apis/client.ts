export const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

/**
 * Standard fetch wrapper that automatically attaches the JWT token
 * from localStorage.
 */
export async function fetchApi(endpoint: string, options: RequestInit = {}) {
    const token = typeof window !== 'undefined' ? localStorage.getItem('wort_token') : null;

    const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        ...(options.headers as Record<string, string>),
    };

    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    // Remove Content-Type if sending FormData (let browser set boundary)
    if (options.body instanceof FormData) {
        delete headers['Content-Type'];
    }

    const res = await fetch(`${API_BASE}${endpoint}`, {
        ...options,
        headers,
    });

    if (!res.ok) {
        let errDetail = res.statusText;
        try {
            const errData = await res.json();
            errDetail = errData.detail || errDetail;
        } catch { }
        throw new Error(errDetail);
    }

    return res.json();
}
