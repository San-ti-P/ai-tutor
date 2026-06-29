"use client";

import { cn } from "@/lib/utils";

interface QuestionNavigatorProps {
  total: number;
  current: number;
  answers: Record<string, string>;
  onSelect: (index: number) => void;
}

export function QuestionNavigator({
  total,
  current,
  answers,
  onSelect,
}: QuestionNavigatorProps) {
  return (
    <div className="flex items-center justify-center gap-2 py-4">
      {Array.from({ length: total }, (_, i) => {
        const isAnswered =
          answers[`q-${i}`] !== undefined && answers[`q-${i}`] !== "";
        const isCurrent = i === current;
        return (
          <button
            key={i}
            type="button"
            onClick={() => onSelect(i)}
            className={cn(
              "flex size-8 items-center justify-center rounded-full text-xs font-medium transition-colors",
              isCurrent
                ? "bg-primary text-primary-foreground"
                : isAnswered
                  ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                  : "bg-muted text-muted-foreground hover:bg-muted/80",
            )}
            title={`Pregunta ${i + 1}${isAnswered ? " (respondida)" : ""}`}
          >
            {i + 1}
          </button>
        );
      })}
    </div>
  );
}
