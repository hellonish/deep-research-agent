import { fetchApi } from './client';
import type { ModelOption } from '@/lib/types';

export interface KeyStatusResponse {
    configured?: boolean;
}

export interface ProviderKeyItem {
    provider: string;
    label: string;
}

export interface ProviderKeysResponse {
    keys: ProviderKeyItem[];
}

export interface AvailableModelsResponse {
    models?: ModelOption[];
}

export interface SetKeyResponse {
    valid: boolean;
    models: Array<Record<string, unknown>>;
    provider?: string;
}

export async function getKeyStatus(): Promise<KeyStatusResponse> {
    return fetchApi('/models/key-status') as Promise<KeyStatusResponse>;
}

export async function getProviderKeys(): Promise<ProviderKeysResponse> {
    return fetchApi('/models/keys') as Promise<ProviderKeysResponse>;
}

export async function getAvailable(): Promise<AvailableModelsResponse> {
    return fetchApi('/models/available') as Promise<AvailableModelsResponse>;
}

export async function setKey(apiKey: string, provider: string = 'gemini'): Promise<SetKeyResponse> {
    return fetchApi('/models/set-key', {
        method: 'POST',
        body: JSON.stringify({ api_key: apiKey, provider }),
    }) as Promise<SetKeyResponse>;
}

export async function setModel(modelId: string): Promise<void> {
    await fetchApi('/models/set-model', {
        method: 'POST',
        body: JSON.stringify({ model_id: modelId }),
    });
}
