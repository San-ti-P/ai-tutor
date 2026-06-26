"use client";

import type { Exam } from "@/lib/types";

interface ExamWidgetProps {
  exam: Exam;
  onAnswerChange?: (questionId: string, value: string) => void;
  onSubmit?: () => void;
}

export function ExamWidget({ exam }: ExamWidgetProps) {
  if (!exam || !exam.questions || exam.questions.length === 0) {
    return (
      <p className="text-muted-foreground text-sm italic">
        El examen no tiene preguntas disponibles.
      </p>
    );
  }

  return (
    <div className="mt-3 rounded-lg border border-border bg-background p-4">
      <h3 className="mb-2 font-semibold text-foreground text-sm">
        Examen: {exam.topic}
      </h3>
      <p className="mb-3 text-muted-foreground text-xs">
        {exam.questions.length} pregunta{exam.questions.length !== 1 && "s"}{" "}
        &middot; Dificultad: {exam.difficulty}
      </p>
      <div className="flex flex-col gap-3">
        {exam.questions.map((q, i) => (
          <div key={q.id} className="rounded-md border border-border p-3">
            <p className="mb-2 font-medium text-foreground text-sm">
              {i + 1}. {q.prompt}
            </p>
            {q.type === "mcq" && q.options ? (
              <div className="flex flex-col gap-1">
                {q.options.map((opt, j) => (
                  <label
                    key={j}
                    className="inline-flex items-center gap-2 text-foreground text-sm"
                  >
                    <input
                      type="radio"
                      name={`exam-question-${q.id}`}
                      value={opt}
                      className="size-3.5 accent-primary"
                    />
                    {opt}
                  </label>
                ))}
              </div>
            ) : (
              <textarea
                placeholder="Escribí tu respuesta acá..."
                rows={3}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-2 focus:ring-ring resize-vertical"
              />
            )}
          </div>
        ))}
      </div>
      {exam.questions.length > 0 && (
        <div className="mt-4 flex justify-end">
          <button
            type="button"
            className="inline-flex items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            Entregar Examen
          </button>
        </div>
      )}
    </div>
  );
}
