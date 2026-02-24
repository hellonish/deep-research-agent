"use client";

import { useAuth } from '@/components/AuthProvider';
import { getProviderKeys, getAvailable, setKey, setModel } from '@/apis';
import type { ModelOption } from '@/lib/types';
import { KeyRound, Cpu, CheckCircle2 } from 'lucide-react';
import { useState, useEffect } from 'react';

const PROVIDER_OPTIONS = [
    { id: 'gemini', label: 'Gemini (Google)' },
    { id: 'deepseek', label: 'DeepSeek' },
    { id: 'openai', label: 'OpenAI' },
];

export default function SettingsPage() {
    const { user } = useAuth();
    const [providerKeys, setProviderKeys] = useState<{ provider: string; label: string }[]>([]);
    const [selectedProvider, setSelectedProvider] = useState<string>('gemini');
    const [apiKey, setApiKey] = useState('');
    const [isSavingKey, setIsSavingKey] = useState(false);
    const [keyStatus, setKeyStatus] = useState<string | null>(null);

    const [models, setModels] = useState<ModelOption[]>([]);
    const [selectedModelId, setSelectedModelId] = useState<string>(user?.selected_model ?? '');
    const [isSavingModel, setIsSavingModel] = useState(false);
    const [modelStatus, setModelStatus] = useState<string | null>(null);

    useEffect(() => {
        if (user?.selected_model) setSelectedModelId(user.selected_model);
    }, [user?.selected_model]);

    const loadKeys = () => {
        getProviderKeys()
            .then((r) => setProviderKeys(r.keys ?? []))
            .catch(() => setProviderKeys([]));
    };

    useEffect(() => {
        loadKeys();
    }, []);

    useEffect(() => {
        getAvailable()
            .then((r) => setModels(r.models ?? []))
            .catch(() => {});
    }, []);

    const handleSaveKey = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!apiKey.trim()) return;
        setIsSavingKey(true);
        setKeyStatus(null);
        try {
            await setKey(apiKey.trim(), selectedProvider);
            loadKeys();
            getAvailable().then((r) => setModels(r.models ?? [])).catch(() => {});
            setKeyStatus(`${PROVIDER_OPTIONS.find(p => p.id === selectedProvider)?.label ?? selectedProvider} key saved.`);
            setApiKey('');
        } catch (err: unknown) {
            setKeyStatus(err instanceof Error ? err.message : 'Failed to save key.');
        } finally {
            setIsSavingKey(false);
        }
    };

    const handleSaveModel = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsSavingModel(true);
        setModelStatus(null);
        try {
            await setModel(selectedModelId);
            setModelStatus('Default model updated.');
        } catch (err: unknown) {
            setModelStatus(err instanceof Error ? err.message : 'Failed to save.');
        } finally {
            setIsSavingModel(false);
        }
    };

    return (
        <div className="p-8 max-w-4xl mx-auto w-full">
            <header className="mb-10">
                <h1 className="text-3xl font-bold mb-2">Settings</h1>
                <p className="text-muted-foreground">API keys (Gemini, DeepSeek, etc.), default model, and preferences.</p>
            </header>

            <div className="grid grid-cols-1 gap-8">
                <section className="glass-panel p-6 rounded-2xl border border-border/50" aria-labelledby="api-keys-heading">
                    <div className="flex items-center gap-3 mb-6">
                        <div className="p-2 bg-primary/15 rounded-lg border border-primary/25">
                            <KeyRound className="w-5 h-5 text-primary" aria-hidden />
                        </div>
                        <h2 id="api-keys-heading" className="text-xl font-semibold">API keys</h2>
                        {providerKeys.length > 0 && (
                            <span className="inline-flex items-center gap-1.5 text-sm text-green-500 font-medium ml-2">
                                <CheckCircle2 className="w-4 h-4 shrink-0" aria-hidden />
                                {providerKeys.map(k => k.label).join(', ')}
                            </span>
                        )}
                    </div>
                    {providerKeys.length > 0 && (
                        <p className="text-sm text-muted-foreground mb-4">
                            You have keys for: {providerKeys.map(k => k.label).join(', ')}. Add or replace below.
                        </p>
                    )}
                    <form onSubmit={handleSaveKey} className="max-w-lg">
                        <label htmlFor="provider-select" className="block text-sm font-medium mb-2 text-muted-foreground">
                            Provider
                        </label>
                        <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
                            <select
                                id="provider-select"
                                value={selectedProvider}
                                onChange={(e) => setSelectedProvider(e.target.value)}
                                className="w-full sm:w-48 bg-secondary border border-border rounded-lg px-4 py-2 text-foreground focus:outline-none focus:border-primary transition-colors"
                            >
                                {PROVIDER_OPTIONS.map((p) => (
                                    <option key={p.id} value={p.id}>{p.label}</option>
                                ))}
                            </select>
                            <input
                                id="api-key-input"
                                type="password"
                                placeholder={selectedProvider === 'gemini' ? 'AIzaSy…' : 'sk-…'}
                                value={apiKey}
                                onChange={(e) => setApiKey(e.target.value)}
                                className="flex-1 min-w-0 bg-secondary border border-border rounded-lg px-4 py-2 focus:outline-none focus:border-primary transition-colors text-foreground"
                                autoComplete="off"
                            />
                            <button
                                type="submit"
                                disabled={isSavingKey || !apiKey.trim()}
                                className="bg-primary text-primary-foreground px-6 py-2 rounded-lg font-medium hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
                            >
                                {isSavingKey ? 'Verifying…' : 'Save'}
                            </button>
                        </div>
                        {keyStatus && (
                            <p className={`mt-2 text-sm ${keyStatus.includes('saved') ? 'text-green-400' : 'text-red-400'}`} role="status">
                                {keyStatus}
                            </p>
                        )}
                        <p className="mt-3 text-sm text-muted-foreground">
                            Stored securely; never shown again after saving.
                        </p>
                    </form>
                </section>

                <section className="glass-panel p-6 rounded-2xl border border-border/50" aria-labelledby="model-heading">
                    <div className="flex items-center gap-3 mb-6">
                        <div className="p-2 bg-primary/15 rounded-lg border border-primary/25">
                            <Cpu className="w-5 h-5 text-primary" aria-hidden />
                        </div>
                        <h2 id="model-heading" className="text-xl font-semibold">Default model</h2>
                    </div>
                    <form onSubmit={handleSaveModel} className="max-w-lg">
                        <label htmlFor="model-select" className="block text-sm font-medium mb-2 text-muted-foreground">
                            Model used for chat and research when none is selected per request
                        </label>
                        <div className="flex gap-3">
                            <select
                                id="model-select"
                                value={selectedModelId}
                                onChange={(e) => setSelectedModelId(e.target.value)}
                                className="flex-1 bg-secondary border border-border rounded-lg px-4 py-2 text-foreground focus:outline-none focus:border-primary transition-colors"
                            >
                                {models.map((m) => (
                                    <option key={m.id} value={m.id}>
                                        {m.provider ? `${m.provider} · ` : ''}{m.display_name || m.name || m.id}
                                    </option>
                                ))}
                            </select>
                            <button
                                type="submit"
                                disabled={isSavingModel}
                                className="bg-primary text-primary-foreground px-6 py-2 rounded-lg font-medium hover:opacity-90 transition-opacity disabled:opacity-50"
                            >
                                {isSavingModel ? 'Saving…' : 'Save'}
                            </button>
                        </div>
                        {modelStatus && (
                            <p className="mt-2 text-sm text-green-400" role="status">{modelStatus}</p>
                        )}
                    </form>
                </section>
            </div>
        </div>
    );
}
