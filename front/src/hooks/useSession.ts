"use client";

import { useState, useCallback, useEffect } from "react";

function generateUUID(): string {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        const v = c === "x" ? r : (r & 0x3) | 0x8;
        return v.toString(16);
      });
}

const SESSION_KEY = "ai-tutor-session-id";

export function useSession() {
  const [sessionId, setSessionId] = useState<string>("");

  useEffect(() => {
    let id = localStorage.getItem(SESSION_KEY);
    if (!id) {
      id = generateUUID();
      localStorage.setItem(SESSION_KEY, id);
    }
    setSessionId(id);
  }, []);

  const resetSession = useCallback(() => {
    const id = generateUUID();
    localStorage.setItem(SESSION_KEY, id);
    setSessionId(id);
  }, []);

  return { sessionId, resetSession };
}
