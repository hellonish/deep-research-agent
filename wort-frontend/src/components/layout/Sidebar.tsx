"use client";

import { useAuth } from '@/components/AuthProvider';
import {
    LogOut,
    Settings,
    Sparkles,
    MessageCircle,
    PenSquare,
    MoreVertical,
    Edit2,
    Trash2,
    Check,
    X,
    PanelLeftClose,
    PanelLeft,
    FileSearch,
} from 'lucide-react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import {
    getChats,
    getResearch,
    deleteChat,
    deleteResearch,
    updateChatTitle,
    updateResearchTitle,
} from '@/apis';

interface HistoryItem {
    id: string;
    sessionId: string;
    title: string;
    type: 'chat' | 'research';
    date: Date;
    href: string;
}

export function Sidebar() {
    const { user, logout, token } = useAuth();
    const pathname = usePathname();
    const router = useRouter();
    const [collapsed, setCollapsed] = useState(false);
    const [history, setHistory] = useState<HistoryItem[]>([]);
    const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [editTitle, setEditTitle] = useState('');

    const fetchHistory = () => {
        if (!token) return;
        Promise.all([getChats(), getResearch()])
            .then(([chatsData, researchData]) => {
                const combined: HistoryItem[] = [];
                chatsData.forEach((c) => {
                    combined.push({
                        id: c.id,
                        sessionId: c.id,
                        title: c.title || 'New Chat',
                        type: 'chat',
                        date: new Date(c.updated_at || c.created_at || 0),
                        href: `/chat/${c.id}`,
                    });
                });
                researchData.forEach((r) => {
                    combined.push({
                        id: r.id,
                        sessionId: r.session_id ?? r.id,
                        title: r.query || 'Research Job',
                        type: 'research',
                        date: new Date(r.created_at || 0),
                        href: `/research/${r.id}`,
                    });
                });
                const uniqueMap = new Map<string, HistoryItem>();
                combined.forEach((item) => {
                    if (!uniqueMap.has(item.sessionId) || item.type === 'research') {
                        uniqueMap.set(item.sessionId, item);
                    }
                });
                const uniqueList = Array.from(uniqueMap.values());
                uniqueList.sort((a, b) => b.date.getTime() - a.date.getTime());
                setHistory(uniqueList);
            })
            .catch((err) => console.error('Failed to load history in sidebar', err));
    };

    useEffect(() => {
        fetchHistory();
    }, [pathname, token]);

    const handleDelete = async (e: React.MouseEvent, id: string, type: string) => {
        e.preventDefault();
        e.stopPropagation();
        if (!confirm('Delete this session? This cannot be undone.')) return;
        try {
            if (type === 'chat') await deleteChat(id);
            else await deleteResearch(id);
            fetchHistory();
            if (pathname.includes(id)) router.push('/chat');
        } catch (err) {
            console.error('Failed to delete history item:', err);
        }
    };

    const handleRename = async (e: React.FormEvent, id: string, type: string) => {
        e.preventDefault();
        if (!editTitle.trim()) {
            setEditingId(null);
            return;
        }
        try {
            if (type === 'chat') await updateChatTitle(id, editTitle.trim());
            else await updateResearchTitle(id, editTitle.trim());
            setEditingId(null);
            fetchHistory();
        } catch (err) {
            console.error('Failed to rename history item:', err);
        }
    };

    const navItems: { name: string; href: string; icon: typeof PenSquare }[] = [
        { name: 'New chat', href: '/chat', icon: PenSquare },
    ];

    const isActive = (href: string) => pathname === href || (href === '/chat' && (pathname === '/chat' || pathname.startsWith('/chat/')));

    return (
        <aside
            className={`border-r border-border bg-card/80 backdrop-blur-xl flex flex-col h-screen shrink-0 transition-[width] duration-200 ${
                collapsed ? 'w-[72px]' : 'w-64'
            }`}
            role="navigation"
            aria-label="Main navigation"
        >
            {/* Logo + collapse */}
            <div className="p-4 flex items-center justify-between shrink-0 border-b border-border/50">
                <Link href="/chat" className="flex items-center gap-3 min-w-0">
                    <div className="p-2 bg-primary/10 rounded-xl border border-primary/20 shrink-0">
                        <Sparkles className="w-6 h-6 text-primary" aria-hidden />
                    </div>
                    {!collapsed && (
                        <span className="font-bold text-xl tracking-tight text-glow truncate">Wort</span>
                    )}
                </Link>
                <button
                    type="button"
                    onClick={() => setCollapsed((c) => !c)}
                    className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
                    aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                >
                    {collapsed ? <PanelLeft className="w-5 h-5" /> : <PanelLeftClose className="w-5 h-5" />}
                </button>
            </div>

            {/* Primary nav */}
            <nav className="px-3 py-4 space-y-1 border-b border-border/50">
                {navItems.map((item) => {
                    const active = isActive(item.href);
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg border transition-all ${
                                active
                                    ? 'bg-primary/15 text-primary border-primary/25'
                                    : 'text-muted-foreground hover:bg-secondary hover:text-foreground border-transparent'
                            } ${collapsed ? 'justify-center px-2' : ''}`}
                        >
                            <item.icon className="w-5 h-5 shrink-0" aria-hidden />
                            {!collapsed && <span className="font-medium">{item.name}</span>}
                        </Link>
                    );
                })}
            </nav>

            {/* Recent */}
            <div className="flex-1 overflow-y-auto px-3 py-4 space-y-1 min-h-0">
                {!collapsed && (
                    <p className="px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                        Recent
                    </p>
                )}
                {history.length > 0 ? (
                    history.map((item) => {
                        const active = pathname === item.href;
                        if (editingId === item.id) {
                            return (
                                <div
                                    key={`${item.type}-${item.id}`}
                                    className={`flex items-center gap-2 rounded-lg bg-secondary/50 ${collapsed ? 'px-2 py-2' : 'px-3 py-2'}`}
                                >
                                    <form
                                        onSubmit={(e) => handleRename(e, item.id, item.type)}
                                        className="flex items-center gap-2 w-full"
                                    >
                                        {!collapsed && (
                                            <input
                                                autoFocus
                                                value={editTitle}
                                                onChange={(e) => setEditTitle(e.target.value)}
                                                className="flex-1 min-w-0 bg-transparent border-b border-primary/50 text-sm focus:outline-none text-foreground"
                                                aria-label="New title"
                                            />
                                        )}
                                        <button
                                            type="button"
                                            onClick={() => setEditingId(null)}
                                            className="text-muted-foreground hover:text-foreground shrink-0 p-1"
                                            aria-label="Cancel"
                                        >
                                            <X className="w-3.5 h-3.5" />
                                        </button>
                                        <button
                                            type="submit"
                                            className="text-green-500/80 hover:text-green-400 shrink-0 p-1"
                                            aria-label="Save"
                                        >
                                            <Check className="w-3.5 h-3.5" />
                                        </button>
                                    </form>
                                </div>
                            );
                        }
                        return (
                            <div key={`${item.type}-${item.id}`} className="relative group">
                                <Link
                                    href={item.href}
                                    className={`flex items-center gap-3 rounded-lg transition-all pr-8 ${
                                        collapsed ? 'px-2 py-2 justify-center' : 'px-3 py-2'
                                    } ${active ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:bg-secondary/50 hover:text-foreground'}`}
                                >
                                    {item.type === 'research' ? (
                                        <FileSearch className="w-4 h-4 shrink-0 text-primary/80" aria-hidden />
                                    ) : (
                                        <MessageCircle className="w-4 h-4 shrink-0 text-primary/70" aria-hidden />
                                    )}
                                    {!collapsed && <span className="text-sm truncate">{item.title}</span>}
                                </Link>
                                {!collapsed && (
                                    <>
                                        <button
                                            type="button"
                                            className={`absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded-md text-muted-foreground hover:bg-background hover:text-foreground transition-all ${
                                                menuOpenId === item.id ? 'opacity-100 flex' : 'opacity-0 hidden group-hover:flex group-hover:opacity-100'
                                            }`}
                                            onClick={(e) => {
                                                e.preventDefault();
                                                setMenuOpenId(menuOpenId === item.id ? null : item.id);
                                            }}
                                            aria-label="Session options"
                                            aria-expanded={menuOpenId === item.id}
                                        >
                                            <MoreVertical className="w-4 h-4" />
                                        </button>
                                        {menuOpenId === item.id && (
                                            <div
                                                className="absolute right-0 top-full mt-1 w-36 bg-card border border-border rounded-lg shadow-xl py-1 z-50"
                                                role="menu"
                                            >
                                                <button
                                                    type="button"
                                                    className="flex items-center gap-2 px-3 py-2 w-full text-left text-sm text-muted-foreground hover:text-foreground hover:bg-secondary"
                                                    onClick={(e) => {
                                                        e.preventDefault();
                                                        setEditingId(item.id);
                                                        setEditTitle(item.title);
                                                        setMenuOpenId(null);
                                                    }}
                                                    role="menuitem"
                                                >
                                                    <Edit2 className="w-3.5 h-3.5" /> Rename
                                                </button>
                                                <button
                                                    type="button"
                                                    className="flex items-center gap-2 px-3 py-2 w-full text-left text-sm text-red-500/90 hover:text-red-400 hover:bg-red-950/30"
                                                    onClick={(e) => {
                                                        handleDelete(e, item.id, item.type);
                                                        setMenuOpenId(null);
                                                    }}
                                                    role="menuitem"
                                                >
                                                    <Trash2 className="w-3.5 h-3.5" /> Delete
                                                </button>
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                        );
                    })
                ) : (
                    !collapsed && (
                        <p className="px-3 text-sm text-muted-foreground italic">No recent sessions</p>
                    )
                )}
            </div>

            {/* User + Settings + Logout */}
            <div className="p-3 border-t border-border shrink-0">
                {!collapsed && (
                    <div className="flex items-center gap-3 px-3 py-2 mb-2">
                        {user?.picture ? (
                            <img
                                src={user.picture}
                                alt=""
                                className="w-8 h-8 rounded-full border border-border shrink-0"
                            />
                        ) : (
                            <div className="w-8 h-8 rounded-full bg-secondary border border-border shrink-0" />
                        )}
                        <p className="text-sm font-medium truncate text-foreground flex-1 min-w-0">
                            {user?.name || 'User'}
                        </p>
                    </div>
                )}
                <Link
                    href="/settings"
                    className={`flex items-center gap-3 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors ${collapsed ? 'justify-center p-2' : 'px-3 py-2'}`}
                >
                    <Settings className="w-4 h-4 shrink-0" aria-hidden />
                    {!collapsed && <span className="text-sm">Settings</span>}
                </Link>
                <button
                    type="button"
                    onClick={logout}
                    className={`w-full flex items-center gap-3 rounded-lg text-red-400 hover:text-red-300 hover:bg-red-950/30 transition-colors text-left mt-1 ${collapsed ? 'justify-center p-2' : 'px-3 py-2'}`}
                >
                    <LogOut className="w-4 h-4 shrink-0" aria-hidden />
                    {!collapsed && <span className="text-sm">Log out</span>}
                </button>
            </div>
        </aside>
    );
}
