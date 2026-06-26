"use client";

import { useState, useCallback, useEffect } from "react";
import { useStudySession } from "@/hooks/useStudySession";
import { api } from "@/lib/api";
import { ChatMessageList } from "@/components/chat/ChatMessageList";
import { ChatInput } from "@/components/chat/ChatInput";
import { UploadDropzone } from "@/components/upload/UploadDropzone";
import { SessionSidebar } from "@/components/layout/SessionSidebar";
import { SessionFileList } from "@/components/upload/SessionFileList";
import {
  UploadFileList,
  type FileStatus,
} from "@/components/upload/UploadFileList";
import type { ChatMessage, IngestResult } from "@/lib/types";

interface FileEntry {
  file: File;
  status: FileStatus;
  message?: string;
  result?: {
    classification?: string;
    topicsDetected?: string[];
    chunksCreated?: number;
  };
}

export default function ChatPage() {
  const {
    sessions,
    activeSession,
    isLoading: sessionsLoading,
    createSession,
    switchSession,
    deleteSession,
    refreshSessions,
  } = useStudySession();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [fileEntries, setFileEntries] = useState<FileEntry[]>([]);

  const sessionId = activeSession?.id ?? "";

  // Auto-create default session if none exist after load
  useEffect(() => {
    if (!sessionsLoading && sessions.length === 0) {
      createSession("General").catch(() => {});
    }
  }, [sessionsLoading, sessions.length, createSession]);

  // Clear messages when switching sessions
  useEffect(() => {
    setMessages([]);
    setFileEntries([]);
  }, [sessionId]);

  const handleFilesSelected = useCallback(async (newFiles: File[]) => {
    if (!sessionId) return;

    const entries: FileEntry[] = newFiles.map((f) => ({
      file: f,
      status: "uploading" as FileStatus,
    }));
    setFileEntries((prev) => [...prev, ...entries]);

    try {
      const res = await api.uploadDocuments(newFiles, sessionId);
      const results = Array.isArray(res.data) ? res.data : [res.data];

      // Refresh sessions to pick up file_count update
      refreshSessions();

      setFileEntries((prev) => {
        const updated = [...prev];
        for (const entry of updated) {
          if (entry.status === "uploading") {
            const idx = newFiles.findIndex(
              (f) => f.name === entry.file.name && f.size === entry.file.size,
            );
            if (idx >= 0 && results[idx]) {
              const r: IngestResult = results[idx];
              const isError =
                r.status === "error" || r.status === "rejected_non_academic";
              entry.status = isError ? "error" : "complete";
              entry.result = {
                classification: r.classification,
                topicsDetected: r.topicsDetected,
                chunksCreated: r.chunksCreated,
              };
              if (r.status === "rejected_non_academic") {
                entry.status = "rejected";
                entry.message = "No es material académico";
              }
            } else {
              entry.status = "complete";
            }
          }
        }
        return updated;
      });
    } catch {
      setFileEntries((prev) =>
        prev.map((e) =>
          e.status === "uploading"
            ? { ...e, status: "error" as FileStatus }
            : e,
        ),
      );
    }
  }, [sessionId, refreshSessions]);

  const handleSend = useCallback(
    async (text: string) => {
      if (!text.trim() || !sessionId) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      try {
        const res = await api.chat({
          session_id: sessionId,
          message: text,
        });

        const assistantMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: res.data.response,
          timestamp: new Date(),
          traceId: res.data.trace_id ?? res.trace_id,
          exam: res.data.exam,
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        const msg =
          err instanceof Error
            ? err.message
            : "No se pudo conectar con el servidor. Verificá que el backend esté corriendo.";
        const errorMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: msg,
          timestamp: new Date(),
          isError: true,
        };
        setMessages((prev) => [...prev, errorMsg]);
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId],
  );

  if (sessionsLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <p className="text-muted-foreground text-sm">Cargando sesiones...</p>
      </div>
    );
  }

  return (
    <div className="flex h-screen">
      <SessionSidebar
        sessions={sessions}
        activeSession={activeSession}
        onCreate={createSession}
        onSwitch={switchSession}
        onDelete={deleteSession}
      />

      <div className="flex flex-col flex-1 gap-6 overflow-y-auto p-6">
        <div className="text-center">
          <h1 className="font-bold text-3xl text-foreground tracking-tight">
            Tutor Académico
          </h1>
          <p className="mt-2 text-muted-foreground">
            {activeSession
              ? `Sesión: ${activeSession.name}`
              : "Prepará tus exámenes con IA"}
          </p>
        </div>

        <div className="flex flex-col gap-2">
          <UploadDropzone
            onFilesSelected={handleFilesSelected}
            activeSessionId={sessionId}
          />
          <SessionFileList sessionId={sessionId} />
          <UploadFileList files={fileEntries} />
        </div>

        <div
          className="flex flex-1 flex-col rounded-lg border border-border bg-card overflow-hidden"
          style={{ minHeight: "60vh" }}
        >
          <ChatMessageList messages={messages} isLoading={isLoading} />
          {!sessionId ? (
            <div className="border-t border-border p-4 text-center text-muted-foreground text-sm">
              Inicializando sesión...
            </div>
          ) : (
            <ChatInput onSend={handleSend} disabled={isLoading} />
          )}
        </div>
      </div>
    </div>
  );
}
