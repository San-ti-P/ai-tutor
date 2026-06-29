"use client";

import { useState, useEffect } from "react";
import { BarChart3 } from "lucide-react";
import { EvaluationView } from "@/components/evaluation/EvaluationView";
import { Spinner } from "@/components/ui/spinner";
import { api } from "@/lib/api";
import { useSessionContext } from "@/hooks/SessionProvider";
import type { ExamEvaluationSummary } from "@/lib/types";

function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleString("es-AR", {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    return iso;
  }
}

function ScoreBadge({ score }: { score: number | null }) {
  if (score === null) return null;
  const pct = Math.round((score / 10) * 100);
  const color =
    pct >= 70
      ? "text-green-600 bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800 dark:text-green-400"
      : pct >= 50
        ? "text-amber-600 bg-amber-50 border-amber-200 dark:bg-amber-950 dark:border-amber-800 dark:text-amber-400"
        : "text-red-600 bg-red-50 border-red-200 dark:bg-red-950 dark:border-red-800 dark:text-red-400";
  return (
    <span className={`rounded-full border px-2.5 py-0.5 text-xs font-medium ${color}`}>
      {score.toFixed(1)} / 10
    </span>
  );
}

export function ResultsContent() {
  const { sessionId } = useSessionContext();
  const [summaries, setSummaries] = useState<ExamEvaluationSummary[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    setIsLoading(true);
    setError(null);
    api
      .getSessionEvaluations(sessionId)
      .then((res) => {
        setSummaries(res.data ?? []);
        if (res.data?.length) setExpanded(res.data[0].examId);
      })
      .catch(() => setError("No se pudieron cargar los resultados."))
      .finally(() => setIsLoading(false));
  }, [sessionId]);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center gap-4 py-16">
        <Spinner size="lg" />
        <p className="text-muted-foreground text-sm">Cargando resultados...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col gap-8">
        <h1 className="font-bold text-3xl text-foreground tracking-tight">Resultados</h1>
        <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-red-700 text-sm dark:border-red-800 dark:bg-red-950 dark:text-red-300">
          {error}
        </div>
      </div>
    );
  }

  if (!summaries || summaries.length === 0) {
    return (
      <div className="flex flex-col gap-8">
        <h1 className="font-bold text-3xl text-foreground tracking-tight">Resultados</h1>
        <div
          data-testid="results-container"
          className="flex flex-col items-center gap-4 rounded-lg border border-border bg-card p-12"
        >
          <BarChart3 className="size-12 text-muted-foreground/40" />
          <p className="text-center text-muted-foreground text-sm">
            Completá un examen para ver tus resultados acá.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div data-testid="results-container" className="flex flex-col gap-6">
      <h1 className="font-bold text-3xl text-foreground tracking-tight">Resultados</h1>

      {summaries.map((summary) => {
        const isOpen = expanded === summary.examId;
        return (
          <div key={summary.examId} className="rounded-lg border border-border bg-card overflow-hidden">
            <button
              type="button"
              className="flex w-full items-center justify-between px-5 py-4 text-left hover:bg-muted/30 transition-colors"
              onClick={() => setExpanded(isOpen ? null : summary.examId)}
            >
              <div className="flex flex-col gap-0.5">
                <span className="font-medium text-foreground text-sm">
                  Examen del {formatDate(summary.createdAt)}
                </span>
                <span className="text-muted-foreground text-xs">
                  {summary.results.length} pregunta{summary.results.length !== 1 ? "s" : ""}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <ScoreBadge score={summary.averageScore} />
                <span className="text-muted-foreground text-sm">{isOpen ? "▲" : "▼"}</span>
              </div>
            </button>

            {isOpen && (
              <div className="border-t border-border px-5 py-4">
                <EvaluationView results={summary.results} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
