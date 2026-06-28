"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage as ChatMessageType, ExamQuestion } from "@/lib/types";
import { ExamWidget } from "@/components/chat/ExamWidget";
import { cn } from "@/lib/utils";

interface ChatMessageProps {
  message: ChatMessageType;
  onExamSubmit?: (examId: string, answers: Record<string, string>, examQuestions: ExamQuestion[]) => void;
}

export function ChatMessage({ message, onExamSubmit }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex w-full gap-3",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      <div
        className={cn(
          "group relative max-w-[80%] rounded-lg px-4 py-2.5 text-sm",
          isUser
            ? "bg-primary text-primary-foreground"
            : message.isError
              ? "bg-red-50 text-red-800 dark:bg-red-950 dark:text-red-200 border border-red-200 dark:border-red-800"
              : "bg-muted text-foreground",
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {!isUser && message.exam && (
          <ExamWidget
            exam={message.exam}
            onSubmit={(answers) => onExamSubmit?.(message.exam!.id, answers, message.exam!.questions)}
          />
        )}

        {message.traceId && (
          <span
            title={`Trace: ${message.traceId}`}
            className="absolute bottom-0.5 right-1 cursor-help text-[10px] text-foreground/30 opacity-0 transition-opacity group-hover:opacity-100"
          >
            {message.traceId.slice(0, 8)}
          </span>
        )}
      </div>
    </div>
  );
}
