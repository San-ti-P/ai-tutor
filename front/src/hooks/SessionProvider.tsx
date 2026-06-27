"use client";

import { createContext, useContext } from "react";
import { useStudySession } from "./useStudySession";
import type { Session, SessionCreate } from "@/lib/types";

interface SessionContextValue {
  sessions: Session[];
  activeSession: Session | null;
  isLoading: boolean;
  sessionId: string;
  studentId: string;
  createSession: (name: string, description?: string) => Promise<Session>;
  renameSession: (
    id: string,
    name: string,
    description?: string,
  ) => Promise<void>;
  switchSession: (id: string) => void;
  deleteSession: (id: string) => Promise<void>;
  refreshSessions: () => Promise<void>;
}

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const hook = useStudySession();

  return (
    <SessionContext.Provider
      value={{
        sessions: hook.sessions,
        activeSession: hook.activeSession,
        isLoading: hook.isLoading,
        studentId: hook.studentId,
        sessionId: hook.activeSession?.id ?? "",
        createSession: hook.createSession,
        renameSession: hook.renameSession,
        switchSession: hook.switchSession,
        deleteSession: hook.deleteSession,
        refreshSessions: hook.refreshSessions,
      }}
    >
      {children}
    </SessionContext.Provider>
  );
}

export function useSessionContext(): SessionContextValue {
  const ctx = useContext(SessionContext);
  if (!ctx) {
    throw new Error("useSessionContext must be used within SessionProvider");
  }
  return ctx;
}
