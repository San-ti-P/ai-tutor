"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { FileText } from "lucide-react";
import { useSessionContext } from "@/hooks/SessionProvider";
import { api } from "@/lib/api";
import { ExamForm } from "@/components/exam/ExamForm";
import { ExamRenderer } from "@/components/exam/ExamRenderer";
import { QuestionNavigator } from "@/components/exam/QuestionNavigator";
import { Spinner } from "@/components/ui/spinner";
import { Button } from "@/components/ui/button";
import type { Exam, Difficulty } from "@/lib/types";

export default function ExamPage() {
  const { sessionId, activeSession } = useSessionContext();
  const router = useRouter();
  const [exam, setExam] = useState<Exam | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showConfirm, setShowConfirm] = useState(false);

  const handleGenerate = useCallback(
    async (data: {
      topic: string;
      difficulty: Difficulty;
      questionCount: number;
      questionTypes: ("mcq" | "open")[];
    }) => {
      if (!sessionId) return;
      setIsGenerating(true);
      setError(null);
      try {
        const res = await api.generateExam({
          session_id: sessionId,
          topic: data.topic,
          preferences: {
            questionTypes: data.questionTypes,
            difficulty: data.difficulty,
            questionCount: data.questionCount,
            includeTopics: [],
            excludeTopics: [],
          },
        });
        setExam(res.data);
        setAnswers({});
        setCurrentIndex(0);
        if (res.data.questions.length === 0) {
          setError(
            "El examen no tiene preguntas. Intentá con otro tema o cargá más material.",
          );
        }
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "No se pudo generar el examen. Intentá de nuevo.",
        );
      } finally {
        setIsGenerating(false);
      }
    },
    [sessionId],
  );

  const handleAnswerChange = useCallback(
    (questionId: string, value: string) => {
      setAnswers((prev) => ({ ...prev, [questionId]: value }));
    },
    [],
  );

  const unanswered = exam
    ? exam.questions.filter((q) => !answers[q.id]).length
    : 0;

  const handleSubmit = useCallback(async () => {
    if (!exam || !sessionId) return;
    setIsSubmitting(true);
    setShowConfirm(false);
    try {
      const res = await api.submitAnswers({
        session_id: sessionId,
        exam_id: exam.id,
        answers,
        examQuestions: exam.questions,
      });
      // Pass evaluation results via session storage
      sessionStorage.setItem("evaluation-results", JSON.stringify(res.data));
      router.push(
        `/results?exam_id=${encodeURIComponent(exam.id)}&topic=${encodeURIComponent(exam.topic)}`,
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "No se pudo enviar el examen. Intentá de nuevo.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }, [exam, sessionId, answers, router]);

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="font-bold text-3xl text-foreground tracking-tight">
          Generar Examen
        </h1>
        <p className="mt-1 text-muted-foreground">
          Creá exámenes personalizados basados en tu material de estudio
          {activeSession && (
            <>
              {" "}
              &middot; Sesión:{" "}
              <span className="font-medium text-foreground">
                {activeSession.name}
              </span>
            </>
          )}
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

      {activeSession?.status === "active" ? (
        !exam ? (
          <div className="flex flex-col gap-6 rounded-lg border border-border bg-card p-6">
            <ExamForm onSubmit={handleGenerate} isLoading={isGenerating} />
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            <QuestionNavigator
              total={exam.questions.length}
              current={currentIndex}
              answers={Object.fromEntries(
                exam.questions.map((q, i) => [`q-${i}`, answers[q.id] ?? ""]),
              )}
              onSelect={setCurrentIndex}
            />

            <ExamRenderer
              key={exam.questions[currentIndex].id}
              question={exam.questions[currentIndex]}
              currentIndex={currentIndex}
              total={exam.questions.length}
              value={answers[exam.questions[currentIndex].id] ?? ""}
              onChange={(v) =>
                handleAnswerChange(exam.questions[currentIndex].id, v)
              }
            />

            <div className="flex items-center justify-between">
              <Button
                variant="ghost"
                onClick={() => setCurrentIndex((i) => Math.max(0, i - 1))}
                disabled={currentIndex === 0}
              >
                Anterior
              </Button>
              {currentIndex < exam.questions.length - 1 ? (
                <Button
                  variant="ghost"
                  onClick={() =>
                    setCurrentIndex((i) =>
                      Math.min(exam.questions.length - 1, i + 1),
                    )
                  }
                >
                  Siguiente
                </Button>
              ) : (
                <Button
                  onClick={() => setShowConfirm(true)}
                  disabled={isSubmitting}
                >
                  {isSubmitting ? (
                    <>
                      <Spinner size="sm" />
                      Enviando...
                    </>
                  ) : (
                    "Entregar Examen"
                  )}
                </Button>
              )}
            </div>
          </div>
        )
      ) : (
        !exam && !isGenerating && (
          <div className="flex items-center gap-2 rounded-lg border border-border bg-accent p-4 text-accent-foreground text-sm">
            <FileText className="size-4 shrink-0" />
            <p>
              Cargá material de estudio primero para generar exámenes
              personalizados.
            </p>
          </div>
        )
      )}

      {/* Confirmation dialog */}
      {showConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="mx-4 w-full max-w-sm rounded-lg border border-border bg-card p-6 shadow-xl">
            <h3 className="mb-2 font-semibold text-foreground">
              Confirmar entrega
            </h3>
            <p className="mb-4 text-muted-foreground text-sm">
              {unanswered > 0
                ? `¿Estás seguro? Tenés ${unanswered} pregunta${unanswered !== 1 ? "s" : ""} sin responder.`
                : "¿Estás seguro de entregar el examen?"}
            </p>
            <div className="flex justify-end gap-2">
              <Button variant="secondary" onClick={() => setShowConfirm(false)}>
                Cancelar
              </Button>
              <Button onClick={handleSubmit}>Entregar</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
