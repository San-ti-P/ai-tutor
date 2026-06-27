"use client";

import { SessionProvider, useSessionContext } from "@/hooks/SessionProvider";
import { SessionSidebar } from "./SessionSidebar";

function AppContent({ children }: { children: React.ReactNode }) {
  const {
    sessions,
    activeSession,
    isLoading,
    createSession,
    renameSession,
    switchSession,
    deleteSession,
  } = useSessionContext();

  if (isLoading) {
    return (
      <div className="flex h-[calc(100vh-3.5rem)] items-center justify-center">
        <p className="text-muted-foreground text-sm">Cargando sesiones...</p>
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100vh-3.5rem)]">
      <SessionSidebar
        sessions={sessions}
        activeSession={activeSession}
        onCreate={createSession}
        onRename={renameSession}
        onSwitch={switchSession}
        onDelete={deleteSession}
      />

      <div className="flex-1 overflow-y-auto">
        <main className="mx-auto max-w-5xl px-4 py-8">{children}</main>
      </div>
    </div>
  );
}

export function ClientLayout({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <AppContent>{children}</AppContent>
    </SessionProvider>
  );
}
