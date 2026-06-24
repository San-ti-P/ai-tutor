"use client";

import { useState, useCallback } from "react";
import { useSession } from "@/hooks/useSession";
import { api } from "@/lib/api";
import { ChatMessageList } from "@/components/chat/ChatMessageList";
import { ChatInput } from "@/components/chat/ChatInput";
import { UploadDropzone } from "@/components/upload/UploadDropzone";
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
  const { sessionId } = useSession();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileEntries, setFileEntries] = useState<FileEntry[]>([]);

  const handleFilesSelected = useCallback(
    async (newFiles: File[]) => {
      const entries: FileEntry[] = newFiles.map((f) => ({
        file: f,
        status: "uploading" as FileStatus,
      }));
      setFileEntries((prev) => [...prev, ...entries]);

      try {
        const res = await api.uploadDocuments(newFiles);
        const results = Array.isArray(res.data) ? res.data : [res.data];

        setFileEntries((prev) => {
          const updated = [...prev];
          for (const entry of updated) {
            if (entry.status === "uploading") {
              // Match result by original index
              const idx = newFiles.findIndex(
                (f) => f.name === entry.file.name && f.size === entry.file.size
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
                  entry.message = "No es material acad\u00e9mico";
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
            e.status === "uploading" ? { ...e, status: "error" as FileStatus } : e
          )
        );
      }
    },
    []
  );

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
      setError(null);

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
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        const msg =
          err instanceof Error
            ? err.message
            : "No se pudo conectar con el servidor. Verific\u00e1 que el backend est\u00e9 corriendo.";
        setError(msg);
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId]
  );

  return (
    <div className="flex flex-col gap-6">
      <div className="text-center">
        <h1 className="font-bold text-3xl text-foreground tracking-tight">
          Tutor Acad&eacute;mico
        </h1>
        <p className="mt-2 text-muted-foreground">
          Prepar&aacute; tus ex&aacute;menes con IA
        </p>
      </div>

      {error && (
        <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-red-700 text-sm dark:border-red-800 dark:bg-red-950 dark:text-red-300">
          {error}
          <button
            type="button"
            onClick={() => setError(null)}
            className="ml-2 underline"
          >
            Descartar
          </button>
        </div>
      )}

      <div className="flex flex-col gap-2">
        <UploadDropzone onFilesSelected={handleFilesSelected} />
        <UploadFileList files={fileEntries} />
      </div>

      <div
        className="flex flex-1 flex-col rounded-lg border border-border bg-card overflow-hidden"
        style={{ minHeight: "60vh" }}
      >
        <ChatMessageList messages={messages} isLoading={isLoading} />
        {!sessionId ? (
          <div className="border-t border-border p-4 text-center text-muted-foreground text-sm">
            Inicializando sesi&oacute;n...
          </div>
        ) : (
          <ChatInput onSend={handleSend} disabled={isLoading} />
        )}
      </div>
    </div>
  );
}
