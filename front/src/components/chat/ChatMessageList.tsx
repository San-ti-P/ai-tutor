"use client";

import { useEffect, useRef } from "react";
import type { ChatMessage as ChatMessageType } from "@/lib/types";
import { ChatMessage } from "./ChatMessage";
import { Spinner } from "@/components/ui/spinner";

interface ChatMessageListProps {
  messages: ChatMessageType[];
  isLoading: boolean;
}

export function ChatMessageList({ messages, isLoading }: ChatMessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

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
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-4">
      {messages.map((msg) => (
        <ChatMessage key={msg.id} message={msg} />
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
