"use client";

import React, { createContext, useContext, useEffect, useState } from 'react';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { useRouter, usePathname } from 'next/navigation';
import { User, AuthResponse } from '@/lib/types';
import { fetchApi } from '@/lib/api';

interface AuthContextType {
    user: User | null;
    token: string | null;
    isLoading: boolean;
    login: (credential: string) => Promise<void>;
    logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const PUBLIC_ROUTES = ['/'];

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [token, setToken] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const router = useRouter();
    const pathname = usePathname();

    useEffect(() => {
        // Check local storage on mount
        const storedToken = localStorage.getItem('wort_token');
        const storedUser = localStorage.getItem('wort_user');

        if (storedToken && storedUser) {
            setToken(storedToken);
            setUser(JSON.parse(storedUser));
        }

        setIsLoading(false);
    }, []);

    useEffect(() => {
        // Route protection
        if (!isLoading) {
            if (!user && !PUBLIC_ROUTES.includes(pathname)) {
                router.push('/');
            } else if (user && pathname === '/') {
                router.push('/chat');
            }
        }
    }, [user, isLoading, pathname, router]);

    const login = async (credential: string) => {
        try {
            setIsLoading(true);
            const res: AuthResponse = await fetchApi('/auth/google', {
                method: 'POST',
                body: JSON.stringify({ id_token: credential }),
            });

            setToken(res.access_token);
            setUser(res.user);
            localStorage.setItem('wort_token', res.access_token);
            localStorage.setItem('wort_user', JSON.stringify(res.user));

            router.push('/chat');
        } catch (error) {
            console.error('Login failed:', error);
            alert('Login failed. Check console.');
        } finally {
            setIsLoading(false);
        }
    };

    const logout = () => {
        setToken(null);
        setUser(null);
        localStorage.removeItem('wort_token');
        localStorage.removeItem('wort_user');
        router.push('/');
    };

    const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || '';

    return (
        <GoogleOAuthProvider clientId={clientId}>
            <AuthContext.Provider value={{ user, token, isLoading, login, logout }}>
                {children}
            </AuthContext.Provider>
        </GoogleOAuthProvider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}
