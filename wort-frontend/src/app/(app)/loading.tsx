export default function AppLoading() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] gap-4" aria-live="polite" aria-busy="true">
      <div className="w-10 h-10 rounded-full border-2 border-primary border-t-transparent animate-spin" />
      <p className="text-sm text-muted-foreground">Loading…</p>
    </div>
  );
}
