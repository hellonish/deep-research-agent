"use client";

import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function NewResearchPage() {
    const router = useRouter();

    useEffect(() => {
        router.replace('/chat');
    }, [router]);

    return (
        <div className="flex items-center justify-center min-h-[200px]">
            <p className="text-sm text-muted-foreground">Redirecting to chat…</p>
        </div>
    );
}
