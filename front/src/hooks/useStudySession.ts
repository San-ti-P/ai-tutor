"use client";

import { useState, useCallback, useEffect } from "react";
import { api } from "@/lib/api";
import type { Session, SessionCreate } from "@/lib/types";

function generateUUID(): string {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        const v = c === "x" ? r : (r & 0x3) | 0x8;
        return v.toString(16);
      });
}

const ACTIVE_SESSION_KEY = "ai-tutor-active-session";
const STUDENT_ID_KEY = "ai-tutor-student-id";

function getOrCreateStudentId(): string {
  if (typeof window === "undefined") return "";
  let id = localStorage.getItem(STUDENT_ID_KEY);
  if (!id) {
    id = generateUUID();
    localStorage.setItem(STUDENT_ID_KEY, id);
  }
  return id;
}

interface UseStudySession {
  sessions: Session[];
  activeSession: Session | null;
  isLoading: boolean;
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

export function useStudySession(): UseStudySession {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);
  const [studentId, setStudentId] = useState<string>("");

  // Bootstrap: load studentId from localStorage (client-only)
  useEffect(() => {
    setStudentId(getOrCreateStudentId());
  }, []);

  const refreshSessions = useCallback(async () => {
    if (!studentId) return;
    try {
      const res = await api.listSessions(studentId);
      setSessions(res.data);
    } catch {
      setSessions([]);
    }
  }, [studentId]);

  // Load sessions and restore active session on mount
  useEffect(() => {
    let cancelled = false;

    const init = async () => {
      // Bootstrap student ID first (client-only)
      const sid = getOrCreateStudentId();
      if (cancelled) return;
      setStudentId(sid);

      setIsLoading(true);

      // Load sessions
      try {
        const res = await api.listSessions(sid);
        if (!cancelled) setSessions(res.data);
      } catch {
        if (!cancelled) setSessions([]);
      }

      if (cancelled) return;

      const savedId = localStorage.getItem(ACTIVE_SESSION_KEY);
      if (savedId) {
        setActiveSessionId(savedId);
      }

      setIsLoading(false);
    };

    init();
    return () => {
      cancelled = true;
    };
  }, []);

  const activeSession = sessions.find((s) => s.id === activeSessionId) ?? null;

  const createSession = useCallback(
    async (name: string, description?: string): Promise<Session> => {
      const req: SessionCreate = {
        name,
        description: description ?? "",
        studentId,
      };
      const res = await api.createSession(req);
      const session = res.data;

      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      localStorage.setItem(ACTIVE_SESSION_KEY, session.id);

      return session;
    },
    [studentId],
  );

  const renameSession = useCallback(
    async (id: string, name: string, description?: string) => {
      await api.renameSession(id, name, description);
      setSessions((prev) =>
        prev.map((s) =>
          s.id === id
            ? { ...s, name, description: description ?? s.description }
            : s,
        ),
      );
    },
    [],
  );

  const switchSession = useCallback((id: string) => {
    setActiveSessionId(id);
    localStorage.setItem(ACTIVE_SESSION_KEY, id);
  }, []);

  const deleteSession = useCallback(
    async (id: string) => {
      await api.deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (activeSessionId === id) {
        const remaining = sessions.filter((s) => s.id !== id);
        const next = remaining[0]?.id ?? "";
        setActiveSessionId(next);
        if (next) {
          localStorage.setItem(ACTIVE_SESSION_KEY, next);
        } else {
          localStorage.removeItem(ACTIVE_SESSION_KEY);
        }
      }
    },
    [activeSessionId, sessions],
  );

  return {
    sessions,
    activeSession,
    isLoading,
    studentId,
    createSession,
    renameSession,
    switchSession,
    deleteSession,
    refreshSessions,
  };
}
