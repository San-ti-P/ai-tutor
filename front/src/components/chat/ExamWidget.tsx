"use client";

import { useState, useCallback } from "react";
import type { Exam } from "@/lib/types";

interface ExamWidgetProps {
  exam: Exam;
  onAnswerChange?: (questionId: string, value: string) => void;
  onSubmit?: (answers: Record<string, string>) => void;
}

export function ExamWidget({
  exam,
  onAnswerChange,
  onSubmit,
}: ExamWidgetProps) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [submitted, setSubmitted] = useState(false);

  const handleChange = useCallback(
    (questionId: string, value: string) => {
      setAnswers((prev) => ({ ...prev, [questionId]: value }));
      onAnswerChange?.(questionId, value);
    },
    [onAnswerChange],
  );

  const handleSubmit = useCallback(() => {
    setSubmitted(true);
    onSubmit?.(answers);
  }, [answers, onSubmit]);

  if (!exam || !exam.questions || exam.questions.length === 0) {
    return (
      <p className="text-muted-foreground text-sm italic">
        El examen no tiene preguntas disponibles.
      </p>
    );
  }

  const unanswered = exam.questions.filter((q) => !answers[q.id]).length;

  return (
    <div className="mt-3 rounded-lg border border-border bg-background p-4">
      <h3 className="mb-2 font-semibold text-foreground text-sm">
        Examen: {exam.topic}
      </h3>
      <p className="mb-3 text-muted-foreground text-xs">
        {exam.questions.length} pregunta{exam.questions.length !== 1 && "s"}{" "}
        &middot; Dificultad: {exam.difficulty}
        {exam.status && exam.status !== "complete" && (
          <> &middot; Estado: {exam.status}</>
        )}
      </p>

      {exam.topicNotFound && exam.topicNotFound.length > 0 && (
        <p className="mb-2 text-amber-600 text-xs">
          Temas no encontrados: {exam.topicNotFound.join(", ")}
        </p>
      )}

      <div className="flex flex-col gap-3">
        {exam.questions.map((q, i) => (
          <div key={q.id} data-testid="exam-question" className="rounded-md border border-border p-3">
            <p className="mb-2 font-medium text-foreground text-sm">
              {i + 1}. {q.prompt}
            </p>
            {q.type === "mcq" && q.options ? (
              <div className="flex flex-col gap-1">
                {q.options.map((opt, j) => (
                  <label
                    key={j}
                    className={`inline-flex items-center gap-2 rounded px-2 py-1 text-foreground text-sm cursor-pointer transition-colors ${
                      answers[q.id] === opt
                        ? "bg-primary/10"
                        : "hover:bg-muted/50"
                    }`}
                  >
                    <input
                      type="radio"
                      value={opt}
                      checked={answers[q.id] === opt}
                      onChange={() => handleChange(q.id, opt)}
                      disabled={submitted}
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
                value={answers[q.id] ?? ""}
                onChange={(e) => handleChange(q.id, e.target.value)}
                disabled={submitted}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-2 focus:ring-ring resize-vertical"
              />
            )}
          </div>
        ))}
      </div>

      {exam.questions.length > 0 && !submitted && (
        <div className="mt-4 flex items-center justify-between">
          <span className="text-muted-foreground text-xs">
            {unanswered > 0
              ? `${unanswered} sin responder`
              : "Todas respondidas"}
          </span>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={unanswered > 0}
            data-testid="submit-exam-btn"
            className="inline-flex items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            Entregar Examen
          </button>
        </div>
      )}

      {submitted && (
        <p className="mt-3 text-center text-green-600 text-sm font-medium">
          ✅ Examen entregado
        </p>
      )}
    </div>
  );
}
