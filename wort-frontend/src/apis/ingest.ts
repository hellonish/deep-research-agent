import { fetchApi } from './client';

export interface UploadResponse {
    file_name?: string;
    chunks_created?: number;
    collection?: string;
}

export async function upload(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return fetchApi('/ingest/upload', {
        method: 'POST',
        body: formData,
    }) as Promise<UploadResponse>;
}
