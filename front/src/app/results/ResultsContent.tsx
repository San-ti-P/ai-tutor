"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { BarChart3 } from "lucide-react";
import { EvaluationView } from "@/components/evaluation/EvaluationView";
import { Spinner } from "@/components/ui/spinner";
import type { EvaluationResult } from "@/lib/types";

export function ResultsContent() {
  const searchParams = useSearchParams();
  const examId = searchParams.get("exam_id");
  const topic = searchParams.get("topic");
  const [results, setResults] = useState<EvaluationResult[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Try to load from session storage (passed from exam page)
    const stored = sessionStorage.getItem("evaluation-results");
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setResults(Array.isArray(parsed) ? parsed : []);
        sessionStorage.removeItem("evaluation-results");
      } catch {
        // Fall through to empty state
      }
    }
    setIsLoading(false);
  }, []);

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
        <div>
          <h1 className="font-bold text-3xl text-foreground tracking-tight">
            Resultados
          </h1>
        </div>
        <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-red-700 text-sm dark:border-red-800 dark:bg-red-950 dark:text-red-300">
          {error}
        </div>
      </div>
    );
  }

  if (!results && examId) {
    return (
      <div className="flex flex-col gap-8">
        <div>
          <h1 className="font-bold text-3xl text-foreground tracking-tight">
            Resultados
          </h1>
          <p className="mt-1 text-muted-foreground">Examen: {examId}</p>
        </div>
        <div className="flex flex-col items-center gap-4 rounded-lg border border-border bg-card p-12">
          <BarChart3 className="size-12 text-muted-foreground/40" />
          <p className="text-center text-muted-foreground text-sm">
            No se encontraron resultados para este examen. Complet\u00e1 un examen
            para ver tus resultados.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="font-bold text-3xl text-foreground tracking-tight">
          Resultados
        </h1>
        <p className="mt-1 text-muted-foreground">
          {examId ? `Examen: ${examId}` : "Revis\u00e1 tu desempe\u00f1o"}
        </p>
      </div>

      {results && results.length > 0 ? (
        <EvaluationView results={results} knownTopics={topic ? [topic] : []} />
      ) : (
        <div className="flex flex-col items-center gap-4 rounded-lg border border-border bg-card p-12">
          <BarChart3 className="size-12 text-muted-foreground/40" />
          <p className="text-center text-muted-foreground text-sm">
            Complet\u00e1 un examen para ver tus resultados ac\u00e1.
          </p>
        </div>
      )}
    </div>
  );
}
