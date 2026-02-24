import { fetchApi } from './client';
import type { ModelOption } from '@/lib/types';

export interface KeyStatusResponse {
    configured?: boolean;
}

export interface AvailableModelsResponse {
    models?: ModelOption[];
}

export async function getKeyStatus(): Promise<KeyStatusResponse> {
    return fetchApi('/models/key-status') as Promise<KeyStatusResponse>;
}

export async function getAvailable(): Promise<AvailableModelsResponse> {
    return fetchApi('/models/available') as Promise<AvailableModelsResponse>;
}

export async function setKey(apiKey: string): Promise<void> {
    await fetchApi('/models/set-key', {
        method: 'POST',
        body: JSON.stringify({ api_key: apiKey }),
    });
}

export async function setModel(modelId: string): Promise<void> {
    await fetchApi('/models/set-model', {
        method: 'POST',
        body: JSON.stringify({ model_id: modelId }),
    });
}
