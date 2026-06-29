"use client";

import { useEffect, useLayoutEffect, useRef } from "react";
import type { ChatMessage as ChatMessageType, ExamEvalSnapshot } from "@/lib/types";
import { ChatMessage } from "./ChatMessage";
import { Spinner } from "@/components/ui/spinner";

interface ChatMessageListProps {
  messages: ChatMessageType[];
  isLoading: boolean;
  hasMore?: boolean;
  isLoadingHistory?: boolean;
  onLoadMore?: () => void;
  sessionId?: string;
  onExamEvaluated?: (messageId: string, snapshot: ExamEvalSnapshot) => void;
}

export function ChatMessageList({
  messages,
  isLoading,
  hasMore = false,
  isLoadingHistory = false,
  onLoadMore,
  sessionId,
  onExamEvaluated,
}: ChatMessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  // Track the scroll height before messages are prepended to restore position
  const prevScrollHeightRef = useRef<number>(0);

  // Auto-scroll to bottom when new messages arrive (not when loading history)
  useEffect(() => {
    if (!isLoadingHistory) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isLoading, isLoadingHistory]);

  // After older messages are prepended, restore scroll position so the view
  // stays anchored to the same message instead of jumping to the top
  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container || !isLoadingHistory) return;
    const delta = container.scrollHeight - prevScrollHeightRef.current;
    container.scrollTop += delta;
  }, [messages, isLoadingHistory]);

  const handleLoadMore = () => {
    if (!onLoadMore || isLoadingHistory) return;
    // Save current scrollHeight before React re-renders with new messages
    prevScrollHeightRef.current = containerRef.current?.scrollHeight ?? 0;
    onLoadMore();
  };

  if (messages.length === 0 && !isLoading) {
    return (
      <div className="flex flex-1 items-center justify-center p-12">
        <p className="text-center text-muted-foreground text-sm">
          Todavía no hay mensajes. Escribí tu consulta abajo.
        </p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex flex-1 flex-col gap-4 overflow-y-auto p-4">
      {/* Load more button — shown at the top when there is older history */}
      {hasMore && (
        <div className="flex justify-center">
          <button
            onClick={handleLoadMore}
            disabled={isLoadingHistory}
            className="flex items-center gap-2 rounded-full border border-border bg-card px-4 py-1.5 text-sm text-muted-foreground shadow-sm transition hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoadingHistory ? (
              <>
                <Spinner size="sm" />
                Cargando...
              </>
            ) : (
              "↑ Cargar mensajes anteriores"
            )}
          </button>
        </div>
      )}

      {messages.map((msg) => (
        <ChatMessage key={msg.id} message={msg} sessionId={sessionId} onExamEvaluated={onExamEvaluated} />
      ))}
      {isLoading && (
        <div className="flex items-center gap-2 self-start rounded-lg bg-muted px-4 py-2.5">
          <Spinner size="sm" />
          <span className="text-muted-foreground text-sm">Escribiendo...</span>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
