"use client";

import { useAuth } from '@/components/AuthProvider';
import { upload } from '@/apis';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, File, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { useState, useRef } from 'react';

export default function IngestPage() {
    const { token, user } = useAuth();

    const [isDragging, setIsDragging] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [fileToUpload, setFileToUpload] = useState<File | null>(null);

    const [result, setResult] = useState<{
        success: boolean;
        message: string;
        details?: string;
    } | null>(null);

    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    };

    const handleFileSelection = (file: File) => {
        setFileToUpload(file);
        setResult(null);
    };

    const handleUpload = async () => {
        if (!fileToUpload) return;

        setUploading(true);
        setResult(null);

        try {
            const data = await upload(fileToUpload);

            setResult({
                success: true,
                message: `Successfully ingested ${data.file_name}`,
                details: `Created ${data.chunks_created} vector chunks in collection: ${data.collection}`,
            });
            setFileToUpload(null);
        } catch (err: any) {
            setResult({
                success: false,
                message: 'Upload failed',
                details: err.message || 'An unknown error occurred.',
            });
        } finally {
            setUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    return (
        <div className="p-8 max-w-4xl mx-auto w-full flex flex-col items-center justify-center min-h-[calc(100vh-4rem)]">
            <div className="text-center mb-10">
                <h1 className="text-3xl font-bold mb-2">Ingest Data</h1>
                <p className="text-muted-foreground max-w-lg mx-auto">
                    Upload PDFs, Word Docs, Text, or HTML files to vectorize them and add them to your personal knowledge base.
                </p>
            </div>

            <div className="w-full max-w-2xl text-center">
                {/* Dropzone */}
                <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className={`glass-panel relative overflow-hidden rounded-3xl border-2 border-dashed transition-all cursor-pointer p-12 flex flex-col items-center justify-center min-h-[300px]
            ${isDragging ? 'border-primary bg-primary/5' : 'border-border/60 hover:border-primary/50'}
            ${uploading ? 'opacity-50 pointer-events-none' : ''}
          `}
                >
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={(e) => e.target.files && handleFileSelection(e.target.files[0])}
                        className="hidden"
                        accept=".pdf,.txt,.md,.html,.htm,.docx"
                    />

                    <AnimatePresence mode="wait">
                        {uploading ? (
                            <motion.div
                                key="uploading"
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.9 }}
                                className="flex flex-col items-center text-primary"
                            >
                                <Loader2 className="w-16 h-16 animate-spin mb-4" />
                                <p className="font-medium text-lg text-glow">Vectorizing Document...</p>
                                <p className="text-sm text-primary/70 mt-2">This may take a moment depending on the file size.</p>
                            </motion.div>
                        ) : fileToUpload ? (
                            <motion.div
                                key="file"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -10 }}
                                className="flex flex-col items-center text-foreground"
                            >
                                <div className="p-4 bg-primary/10 rounded-full mb-4 border border-primary/20">
                                    <File className="w-12 h-12 text-primary" />
                                </div>
                                <p className="font-semibold text-xl mb-1">{fileToUpload.name}</p>
                                <p className="text-sm text-muted-foreground mb-6">
                                    {(fileToUpload.size / (1024 * 1024)).toFixed(2)} MB
                                </p>

                                <div className="flex gap-4">
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setFileToUpload(null);
                                            if (fileInputRef.current) fileInputRef.current.value = '';
                                        }}
                                        className="px-6 py-2 rounded-full border border-border hover:bg-secondary transition-colors"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            handleUpload();
                                        }}
                                        className="px-6 py-2 rounded-full bg-primary text-primary-foreground font-semibold hover:bg-primary/90 transition-colors shadow-lg shadow-primary/20"
                                    >
                                        Upload & Process
                                    </button>
                                </div>
                            </motion.div>
                        ) : (
                            <motion.div
                                key="prompt"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="flex flex-col items-center text-muted-foreground"
                            >
                                <div className="p-4 bg-secondary/50 rounded-full mb-6 ring-8 ring-background/50">
                                    <UploadCloud className={`w-12 h-12 transition-colors ${isDragging ? 'text-primary' : 'text-muted-foreground'}`} />
                                </div>
                                <p className="text-xl font-medium text-foreground mb-2">
                                    Drag & Drop to Upload
                                </p>
                                <p className="text-sm">
                                    or <span className="text-primary hover:underline cursor-pointer">click to browse</span>
                                </p>
                                <div className="flex gap-2 mt-8 text-xs font-mono opacity-50">
                                    <span className="bg-secondary px-2 py-1 rounded">PDF</span>
                                    <span className="bg-secondary px-2 py-1 rounded">DOCX</span>
                                    <span className="bg-secondary px-2 py-1 rounded">TXT</span>
                                    <span className="bg-secondary px-2 py-1 rounded">HTML</span>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                {/* Result Feedback */}
                <AnimatePresence>
                    {result && !uploading && !fileToUpload && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className={`mt-8 p-6 rounded-2xl border text-left flex gap-4 items-start ${result.success ? 'bg-green-500/10 border-green-500/20' : 'bg-red-500/10 border-red-500/20'
                                }`}
                        >
                            {result.success ? (
                                <CheckCircle2 className="w-6 h-6 text-green-500 shrink-0 mt-0.5" />
                            ) : (
                                <XCircle className="w-6 h-6 text-red-500 shrink-0 mt-0.5" />
                            )}
                            <div>
                                <h4 className={`font-semibold text-lg mb-1 ${result.success ? 'text-green-400' : 'text-red-400'}`}>
                                    {result.message}
                                </h4>
                                {result.details && (
                                    <p className="text-sm text-muted-foreground">{result.details}</p>
                                )}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
