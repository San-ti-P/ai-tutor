"use client";

import type { ExamQuestion } from "@/lib/types";
import { Textarea } from "@/components/ui/textarea";

interface ExamRendererProps {
  question: ExamQuestion;
  currentIndex: number;
  total: number;
  value: string;
  onChange: (value: string) => void;
}

export function ExamRenderer({
  question,
  currentIndex,
  total,
  value,
  onChange,
}: ExamRendererProps) {
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-lg border border-border bg-card p-6">
        <div className="mb-4 flex items-center justify-between">
          <span className="font-semibold text-foreground text-sm">
            Pregunta {currentIndex + 1} de {total}
          </span>
          <span className="rounded-full bg-muted px-2 py-0.5 text-muted-foreground text-xs">
            {question.type === "mcq" ? "Multiple choice" : "Respuesta libre"}
          </span>
        </div>

        <h3 className="mb-4 font-medium text-foreground text-lg">
          {question.prompt}
        </h3>

        {question.type === "mcq" && question.options ? (
          <div className="flex flex-col gap-2">
            {question.options.map((opt, i) => (
              <label
                key={i}
                className={`inline-flex items-center gap-3 rounded-md border p-3 text-foreground text-sm cursor-pointer transition-colors ${
                  value === opt
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50"
                }`}
              >
                <input
                  type="radio"
                  name={`q-${question.id}`}
                  value={opt}
                  checked={value === opt}
                  onChange={(e) => onChange(e.target.value)}
                  className="size-3.5 accent-primary"
                />
                {opt}
              </label>
            ))}
            {!value && (
              <p className="mt-1 text-muted-foreground text-xs italic">
                Seleccion&aacute; una respuesta
              </p>
            )}
          </div>
        ) : (
          <Textarea
            rows={5}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="Escrib&iacute; tu respuesta ac&aacute;..."
          />
        )}
      </div>
    </div>
  );
}
