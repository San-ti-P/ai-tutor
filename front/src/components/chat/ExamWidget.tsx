"use client";

import { useState, useCallback } from "react";
import type { Exam, ExamEvalSnapshot } from "@/lib/types";
import { api } from "@/lib/api";

interface ExamWidgetProps {
  exam: Exam;
  sessionId?: string;
  onAnswerChange?: (questionId: string, value: string) => void;
  onEvaluated?: (snapshot: ExamEvalSnapshot) => void;
}

export function ExamWidget({ exam, sessionId, onAnswerChange, onEvaluated }: ExamWidgetProps) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = useCallback(
    (questionId: string, value: string) => {
      setAnswers((prev) => ({ ...prev, [questionId]: value }));
      onAnswerChange?.(questionId, value);
    },
    [onAnswerChange],
  );

  const handleSubmit = useCallback(async () => {
    if (!sessionId) return;
    setIsEvaluating(true);
    setError(null);
    try {
      const res = await api.submitAnswers({
        session_id: sessionId,
        exam_id: exam.id,
        answers,
        examQuestions: exam.questions,
      });

      const results = res.data.map((r) => {
        const qId = r.questionId;
        const q = exam.questions.find((q) => (q.id || "") === qId);
        return {
          ...r,
          questionText: q?.prompt ?? "",
          userAnswer: answers[qId] ?? "",
        };
      });

      onEvaluated?.({ examId: exam.id, topic: exam.topic, results });
    } catch {
      setError("No se pudo obtener la corrección. Intentá de nuevo.");
      setIsEvaluating(false);
    }
  }, [answers, sessionId, exam, onEvaluated]);

  if (!exam?.questions?.length) {
    return (
      <p className="text-muted-foreground text-sm italic">
        El examen no tiene preguntas disponibles.
      </p>
    );
  }

  const unanswered = exam.questions.filter((q, i) => !answers[q.id || `q-${i}`]).length;

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
        {exam.questions.map((q, i) => {
          const qId = q.id || `q-${i}`;
          return (
            <div key={qId} data-testid="exam-question" className="rounded-md border border-border p-3">
              <p className="mb-2 font-medium text-foreground text-sm">
                {i + 1}. {q.prompt}
              </p>
              {q.type === "mcq" && q.options ? (
                <div className="flex flex-col gap-1">
                  {q.options.map((opt, j) => (
                    <label
                      key={j}
                      className={`inline-flex items-center gap-2 rounded px-2 py-1 text-foreground text-sm cursor-pointer transition-colors ${
                        answers[qId] === opt ? "bg-primary/10" : "hover:bg-muted/50"
                      }`}
                    >
                      <input
                        type="radio"
                        value={opt}
                        checked={answers[qId] === opt}
                        onChange={() => handleChange(qId, opt)}
                        disabled={isEvaluating}
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
                  value={answers[qId] ?? ""}
                  onChange={(e) => handleChange(qId, e.target.value)}
                  disabled={isEvaluating}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-2 focus:ring-ring resize-vertical"
                />
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-4 flex items-center justify-between">
        {isEvaluating ? (
          <span className="text-muted-foreground text-sm">Corrigiendo...</span>
        ) : (
          <>
            <span className="text-muted-foreground text-xs">
              {unanswered > 0 ? `${unanswered} sin responder` : "Todas respondidas"}
            </span>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={unanswered > 0 || !sessionId}
              data-testid="submit-exam-btn"
              className="inline-flex items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            >
              Entregar Examen
            </button>
          </>
        )}
      </div>

      {error && <p className="mt-2 text-red-600 text-sm">{error}</p>}
    </div>
  );
}
