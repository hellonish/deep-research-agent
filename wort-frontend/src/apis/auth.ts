import type { AuthResponse } from '@/lib/types';
import { fetchApi } from './client';

export async function googleAuth(idToken: string): Promise<AuthResponse> {
    return fetchApi('/auth/google', {
        method: 'POST',
        body: JSON.stringify({ id_token: idToken }),
    }) as Promise<AuthResponse>;
}
