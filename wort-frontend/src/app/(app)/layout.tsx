import { Sidebar } from '@/components/layout/Sidebar';

export const metadata = {
  title: 'Wort — Chat & Research',
  description: 'Chat, deep research, and document ingestion.',
};

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen min-h-0 bg-background text-foreground">
      <Sidebar />
      <main id="main-content" className="flex-1 flex flex-col min-h-0 overflow-auto" role="main">
        {children}
      </main>
    </div>
  );
}
