"use client";

import { useAuth } from '@/components/AuthProvider';
import { GoogleLogin } from '@react-oauth/google';
import { motion } from 'framer-motion';
import { Sparkles, DatabaseZap, SearchCode } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function LandingPage() {
  const { login, user, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (user && !isLoading) {
      router.push('/chat');
    }
  }, [user, isLoading, router]);

  if (isLoading) return null;

  return (
    <div className="flex flex-col items-center justify-center min-h-screen relative overflow-hidden">
      {/* Background glow effects */}
      <div className="absolute top-1/4 -left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-[100px]" />
      <div className="absolute bottom-1/4 -right-1/4 w-96 h-96 bg-primary/20 rounded-full blur-[100px]" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="glass-panel p-12 rounded-3xl z-10 max-w-2xl w-full text-center border border-border/50"
      >
        <div className="flex justify-center mb-8">
          <div className="p-4 bg-primary/15 rounded-2xl border border-primary/25">
            <Sparkles className="w-16 h-16 text-primary" aria-hidden />
          </div>
        </div>

        <h1 className="text-5xl font-extrabold tracking-tight mb-4 text-glow text-foreground">
          Wort
        </h1>
        <p className="text-xl text-muted-foreground mb-12">
          AI-powered deep research and reasoning.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12 text-left">
          <div className="p-6 rounded-xl bg-secondary/30 border border-border/50">
            <DatabaseZap className="w-8 h-8 text-primary mb-4" aria-hidden />
            <h3 className="font-semibold text-lg mb-2 text-foreground">Ingest</h3>
            <p className="text-sm text-muted-foreground">Vectorize PDFs, docs, and web pages for retrieval.</p>
          </div>
          <div className="p-6 rounded-xl bg-secondary/30 border border-border/50">
            <SearchCode className="w-8 h-8 text-primary mb-4" aria-hidden />
            <h3 className="font-semibold text-lg mb-2 text-foreground">Deep research</h3>
            <p className="text-sm text-muted-foreground">Multi-step research with planning and synthesis.</p>
          </div>
        </div>

        <div className="flex flex-col items-center justify-center gap-4">
          <p className="text-sm text-muted-foreground font-medium mb-2">Sign in to initialize terminal</p>
          <GoogleLogin
            onSuccess={(response) => {
              if (response.credential) {
                login(response.credential);
              }
            }}
            onError={() => {
              console.error('Login Failed');
            }}
            theme="filled_black"
            shape="pill"
          />
        </div>
      </motion.div>
    </div>
  );
}
