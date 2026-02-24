export { API_BASE, fetchApi } from './client';
export { googleAuth } from './auth';
export {
    streamChat,
    startResearch,
    getResearchResult,
    researchScope,
    streamResearchProgress,
    type StartResearchBody,
    type StartResearchResponse,
    type ResearchResultResponse,
    type StreamChatBody,
    type PlanStepDto,
    type ResearchScopeRequest,
    type ResearchScopeResponse,
    type ResearchProgressEvent,
} from './chat';
export {
    getChats,
    getResearch,
    getChatMessages,
    deleteChat,
    deleteResearch,
    updateChatTitle,
    updateResearchTitle,
    type ChatSessionItem,
    type ResearchItem,
} from './history';
export { upload, type UploadResponse } from './ingest';
export {
    getKeyStatus,
    getProviderKeys,
    getAvailable,
    setKey,
    setModel,
    type KeyStatusResponse,
    type ProviderKeysResponse,
    type ProviderKeyItem,
    type AvailableModelsResponse,
} from './models';
