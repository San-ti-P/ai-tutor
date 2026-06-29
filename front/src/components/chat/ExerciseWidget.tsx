"use client";

import { useState } from "react";
import type { Exercise, EvaluationResult } from "@/lib/types";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  ChevronDown,
  ChevronRight,
  CheckCircle,
  HelpCircle,
  Info,
  AlertTriangle,
  Lightbulb,
  Loader2,
} from "lucide-react";

interface ExerciseWidgetProps {
  exercise: Exercise;
  sessionId?: string;
}

export function ExerciseWidget({ exercise, sessionId }: ExerciseWidgetProps) {
  const [showSolution, setShowSolution] = useState(false);
  const [studentAnswer, setStudentAnswer] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [evaluationResult, setEvaluationResult] = useState<EvaluationResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  if (!exercise) return null;

  // Defensive extraction for steps and properties (supports both snake_case and camelCase from backend serialization)
  const statement = exercise.statement || "";
  const givenData = exercise.given_data || (exercise as any).givenData || "";
  const question = exercise.question || "";
  const topics = exercise.topics_covered || (exercise as any).topicsCovered || [];
  
  const modelSolution = exercise.model_solution || (exercise as any).modelSolution || {};
  const steps = modelSolution.steps || [];
  const finalAnswer = modelSolution.finalAnswer || modelSolution.final_answer || "";
  const keyConcepts = modelSolution.keyConcepts || modelSolution.key_concepts || [];

  const handleSubmit = async () => {
    if (!sessionId) {
      setErrorMessage("No hay una sesión activa para realizar la evaluación.");
      return;
    }
    setIsSubmitting(true);
    setErrorMessage(null);
    try {
      const examQuestion = {
        id: exercise.exercise_id,
        type: "open" as const,
        prompt: `${statement}\n\nConsigna: ${question}`,
        baseAnswer: finalAnswer || "",
        topic: topics[0] || "",
        difficulty: exercise.difficulty || "medium",
        sourceChunkIds: exercise.source_chunk_ids || modelSolution.sourceChunkIds || [],
      };

      const res = await api.submitAnswers({
        session_id: sessionId,
        exam_id: `exercise-${exercise.exercise_id}`,
        answers: { [exercise.exercise_id]: studentAnswer },
        examQuestions: [examQuestion],
      });

      if (res.data && res.data.length > 0) {
        setEvaluationResult(res.data[0]);
      } else {
        setErrorMessage("No se recibió respuesta del evaluador.");
      }
    } catch (err) {
      console.error(err);
      setErrorMessage(
        err instanceof Error
          ? err.message
          : "Ocurrió un error al enviar la evaluación."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    setStudentAnswer("");
    setEvaluationResult(null);
    setErrorMessage(null);
  };

  return (
    <div className="mt-3 rounded-lg border border-border bg-background p-4 shadow-sm">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-3">
        <div className="flex items-center gap-2">
          <span className="flex size-6 items-center justify-center rounded bg-primary/10 text-primary">
            <HelpCircle className="size-4" />
          </span>
          <h3 className="font-semibold text-foreground text-sm">
            Ejercicio Práctico
          </h3>
        </div>
        {topics.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {topics.map((t: string, idx: number) => (
              <span
                key={idx}
                className="inline-flex items-center rounded-full bg-muted px-2 py-0.5 text-muted-foreground text-xs"
              >
                {t.split("/").pop() ?? t}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Warnings / Topic suggestions */}
      {exercise.topic_not_found && exercise.topic_not_found.length > 0 && (
        <div className="mt-3 rounded-md bg-amber-500/10 p-3 text-amber-700 dark:text-amber-400">
          <div className="flex gap-2">
            <Info className="size-4 shrink-0 mt-0.5" />
            <div className="text-xs">
              <p className="font-semibold">
                Algunos temas solicitados no fueron encontrados en el material:
              </p>
              <p className="mt-1">{exercise.topic_not_found.join(", ")}</p>
              {exercise.topic_suggestions && exercise.topic_suggestions.length > 0 && (
                <>
                  <p className="mt-2 font-semibold">Temas sugeridos:</p>
                  <p className="mt-0.5">{exercise.topic_suggestions.join(", ")}</p>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Statement / Enunciado */}
      <div className="mt-3 space-y-3">
        <div>
          <h4 className="font-semibold text-foreground text-xs uppercase tracking-wider text-muted-foreground/80">
            Enunciado
          </h4>
          <p className="mt-1 text-sm text-foreground leading-relaxed whitespace-pre-wrap">
            {statement}
          </p>
        </div>

        {/* Given Data / Datos del problema */}
        {givenData && (
          <div className="rounded-md bg-muted/40 p-3">
            <h4 className="font-semibold text-foreground text-xs uppercase tracking-wider text-muted-foreground/80">
              Datos Proporcionados
            </h4>
            <p className="mt-1 text-sm text-foreground font-mono whitespace-pre-wrap">
              {givenData}
            </p>
          </div>
        )}

        {/* Question / Pregunta */}
        <div className="rounded-md border border-primary/20 bg-primary/5 p-3">
          <h4 className="font-semibold text-primary text-xs uppercase tracking-wider">
            Consigna a Resolver
          </h4>
          <p className="mt-1 font-medium text-foreground text-sm leading-relaxed">
            {question}
          </p>
        </div>
      </div>

      {/* Form / Resolution Input */}
      {!evaluationResult && (
        <div className="mt-4 border-t border-border pt-4 space-y-3">
          <label className="block text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Tu Resolución
          </label>
          <textarea
            value={studentAnswer}
            onChange={(e) => setStudentAnswer(e.target.value)}
            disabled={isSubmitting}
            placeholder="Escribí acá tu procedimiento y respuesta para que el tutor la evalúe..."
            className="w-full min-h-[100px] rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
          />
          {errorMessage && (
            <p className="text-xs text-destructive">{errorMessage}</p>
          )}
          <div className="flex justify-end">
            <button
              type="button"
              onClick={handleSubmit}
              disabled={isSubmitting || !studentAnswer.trim()}
              className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Evaluando...
                </>
              ) : (
                "Enviar resolución"
              )}
            </button>
          </div>
        </div>
      )}

      {/* Evaluation Results */}
      {evaluationResult && (
        <div className="mt-4 border-t border-border pt-4 space-y-4 animate-in fade-in duration-300">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              Evaluación del Tutor
            </h4>
            <button
              type="button"
              onClick={handleReset}
              className="text-xs font-medium text-primary hover:underline"
            >
              Intentar de nuevo
            </button>
          </div>

          {evaluationResult.isEvaluable === false ? (
            <div className="rounded-md bg-amber-500/10 p-3 text-amber-700 dark:text-amber-400">
              <div className="flex gap-2">
                <AlertTriangle className="size-4 shrink-0 mt-0.5" />
                <div className="text-xs">
                  <p className="font-semibold">La respuesta no pudo ser evaluada:</p>
                  <p className="mt-1">{evaluationResult.justification || "No es coherente o legible."}</p>
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* Score Badge & Overview */}
              <div className="flex items-start gap-3 rounded-lg border p-4 bg-card shadow-sm">
                <div
                  className={cn(
                    "flex size-12 shrink-0 items-center justify-center rounded-full font-bold text-lg",
                    evaluationResult.score >= 7
                      ? "bg-green-100 text-green-700 dark:bg-green-950/30 dark:text-green-400"
                      : evaluationResult.score >= 4
                        ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-950/30 dark:text-yellow-400"
                        : "bg-red-100 text-red-700 dark:bg-red-950/30 dark:text-red-400"
                  )}
                >
                  {evaluationResult.score}/10
                </div>
                <div className="space-y-1">
                  <p className="font-semibold text-sm text-foreground">
                    {evaluationResult.score >= 7
                      ? "¡Buen trabajo!"
                      : evaluationResult.score >= 4
                        ? "Aprobado con observaciones"
                        : "Requiere revisión"}
                  </p>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {evaluationResult.justification}
                  </p>
                </div>
              </div>

              {/* Conceptual Errors */}
              {evaluationResult.conceptualErrors && evaluationResult.conceptualErrors.length > 0 && (
                <div className="space-y-2">
                  <h5 className="flex items-center gap-1.5 text-xs font-semibold text-destructive uppercase tracking-wider">
                    <AlertTriangle className="size-3.5" />
                    Errores Conceptuales
                  </h5>
                  <ul className="list-inside list-disc pl-2 space-y-1 text-sm text-foreground">
                    {evaluationResult.conceptualErrors.map((error, idx) => (
                      <li key={idx} className="leading-normal">
                        {error}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Suggestions */}
              {evaluationResult.suggestions && evaluationResult.suggestions.length > 0 && (
                <div className="space-y-2">
                  <h5 className="flex items-center gap-1.5 text-xs font-semibold text-primary uppercase tracking-wider">
                    <Lightbulb className="size-3.5" />
                    Sugerencias de Estudio
                  </h5>
                  <ul className="list-inside list-disc pl-2 space-y-1 text-sm text-foreground">
                    {evaluationResult.suggestions.map((suggestion, idx) => (
                      <li key={idx} className="leading-normal">
                        {suggestion}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Collapsible Model Solution */}
      <div className="mt-4 border-t border-border pt-4">
        <button
          type="button"
          onClick={() => setShowSolution(!showSolution)}
          className="inline-flex w-full items-center justify-between rounded-md border border-border bg-muted/30 px-3 py-2 text-left text-sm text-foreground hover:bg-muted/60 transition-colors"
        >
          <span className="font-medium">
            {showSolution ? "Ocultar solución modelo" : "Ver solución modelo"}
          </span>
          {showSolution ? (
            <ChevronDown className="size-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="size-4 text-muted-foreground" />
          )}
        </button>

        {showSolution && (
          <div className="mt-4 space-y-4 border-l-2 border-primary/20 pl-4 animate-in fade-in slide-in-from-left-1 duration-200">
            {/* Steps */}
            {steps.length > 0 ? (
              <div className="space-y-3">
                <h5 className="font-semibold text-foreground text-xs uppercase tracking-wider text-muted-foreground/80">
                  Resolución Paso a Paso
                </h5>
                {steps.map((step: any, idx: number) => {
                  const stepNum = step.stepNumber ?? step.step_number ?? (idx + 1);
                  const desc = step.description || "";
                  const res = step.result || "";

                  return (
                    <div key={idx} className="space-y-1">
                      <div className="flex items-center gap-1.5">
                        <span className="flex size-4 items-center justify-center rounded-full bg-primary/10 font-bold text-primary text-[10px]">
                          {stepNum}
                        </span>
                        <span className="font-medium text-foreground text-xs">
                          Paso {stepNum}
                        </span>
                      </div>
                      <div className="pl-5 text-sm">
                        <p className="text-foreground leading-normal">{desc}</p>
                        {res && (
                          <div className="mt-1 rounded bg-muted px-2 py-1 text-xs font-mono inline-block">
                            Resultado: <span className="text-foreground font-semibold">{res}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-muted-foreground text-xs italic">
                No hay pasos de resolución detallados.
              </p>
            )}

            {/* Final Answer */}
            {finalAnswer && (
              <div className="rounded-md border border-green-500/20 bg-green-500/5 p-3">
                <div className="flex items-start gap-2">
                  <CheckCircle className="size-4 text-green-600 dark:text-green-400 mt-0.5 shrink-0" />
                  <div>
                    <h5 className="font-semibold text-green-800 dark:text-green-400 text-xs uppercase tracking-wider">
                      Respuesta Final
                    </h5>
                    <p className="mt-1 font-medium text-foreground text-sm font-mono leading-relaxed">
                      {finalAnswer}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Key Concepts */}
            {keyConcepts.length > 0 && (
              <div>
                <h5 className="font-semibold text-foreground text-xs uppercase tracking-wider text-muted-foreground/80">
                  Conceptos Clave
                </h5>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {keyConcepts.map((concept: string, idx: number) => (
                    <span
                      key={idx}
                      className="inline-flex items-center rounded bg-primary/5 border border-primary/10 px-2 py-0.5 text-primary text-xs font-medium"
                    >
                      {concept}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

