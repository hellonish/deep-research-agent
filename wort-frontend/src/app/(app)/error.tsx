'use client';

import { useEffect } from 'react';
import { AlertCircle } from 'lucide-react';
import Link from 'next/link';

export default function AppError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('App error:', error);
  }, [error]);

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] px-4 text-center">
      <div className="p-4 rounded-full bg-red-500/10 border border-red-500/20 mb-4">
        <AlertCircle className="w-12 h-12 text-red-400" aria-hidden />
      </div>
      <h1 className="text-xl font-semibold text-foreground mb-2">Something went wrong</h1>
      <p className="text-muted-foreground max-w-md mb-6">
        We couldn’t load this page. You can try again or go back to chat.
      </p>
      <div className="flex gap-3">
        <button
          type="button"
          onClick={reset}
          className="px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium hover:opacity-90 transition-opacity"
        >
          Try again
        </button>
        <Link
          href="/chat"
          className="px-4 py-2 rounded-lg border border-border bg-secondary text-foreground font-medium hover:bg-secondary/80 transition-colors"
        >
          Back to chat
        </Link>
      </div>
    </div>
  );
}
