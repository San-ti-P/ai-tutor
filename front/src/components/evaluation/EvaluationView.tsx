"use client";

import { useState } from "react";
import Link from "next/link";
import type { EvaluationResult } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface EvaluationViewProps {
  results: EvaluationResult[];
  totalScore?: number;
  maxScore?: number;
  knownTopics?: string[];
}

function scoreColor(score: number): "success" | "warning" | "error" {
  if (score >= 7) return "success";
  if (score >= 5) return "warning";
  return "error";
}

function renderSuggestionWithLinks(
  text: string,
  knownTopics: string[],
  index: number,
): React.ReactNode {
  if (knownTopics.length === 0) {
    return text;
  }

  // Find known topics that appear in the suggestion text (case-insensitive)
  const matchingTopics = knownTopics.filter((t) =>
    text.toLowerCase().includes(t.toLowerCase()),
  );

  if (matchingTopics.length === 0) {
    return text;
  }

  // Build a regex that matches any of the matching topics (case-insensitive)
  const escaped = matchingTopics.map((t) =>
    t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
  );
  const pattern = new RegExp(`(${escaped.join("|")})`, "gi");
  const parts = text.split(pattern);

  return parts.map((part, i) => {
    const isTopic = matchingTopics.some(
      (t) => t.toLowerCase() === part.toLowerCase(),
    );
    if (isTopic) {
      return (
        <Link
          key={`${index}-link-${i}`}
          href={`/exam?topic=${encodeURIComponent(part)}`}
          className="underline decoration-amber-500/40 underline-offset-2 hover:decoration-amber-500"
        >
          {part}
        </Link>
      );
    }
    return part;
  });
}

function QuestionCard({
  result: r,
  index: i,
  knownTopics,
}: {
  result: EvaluationResult;
  index: number;
  knownTopics: string[];
}) {
  const [chunksOpen, setChunksOpen] = useState(false);
  const chunks = r.sourceChunks?.filter(Boolean) ?? [];

  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="mb-3 flex items-center justify-between">
        <span className="font-medium text-foreground text-sm">Pregunta {i + 1}</span>
        {r.isEvaluable !== false ? (
          <Badge variant={scoreColor(r.score)}>{r.score.toFixed(0)} / 10</Badge>
        ) : (
          <Badge variant="default">No evaluable</Badge>
        )}
      </div>

      {r.questionText && (
        <p className="mb-2 font-medium text-foreground text-sm">{r.questionText}</p>
      )}

      {r.userAnswer && (
        <div className="mb-3 rounded-md bg-muted/50 px-3 py-2">
          <p className="mb-0.5 font-medium text-muted-foreground text-xs uppercase">Tu respuesta</p>
          <p className="text-foreground text-sm whitespace-pre-wrap">{r.userAnswer}</p>
        </div>
      )}

      {r.justification && (
        <p className="mb-3 text-foreground text-sm">{r.justification}</p>
      )}

      {r.nonEvaluableReason && (
        <p className="mb-3 text-muted-foreground text-sm italic">{r.nonEvaluableReason}</p>
      )}

      {r.conceptualErrors.length > 0 && (
        <div className="mb-2">
          <h4 className="mb-1 font-medium text-red-600 dark:text-red-400 text-xs uppercase">
            Errores conceptuales
          </h4>
          <ul className="flex flex-col gap-1">
            {r.conceptualErrors.map((err, j) => (
              <li key={j} className="text-red-700 dark:text-red-300 text-sm">{err}</li>
            ))}
          </ul>
        </div>
      )}

      {r.suggestions.length > 0 && (
        <div className="mb-3">
          <h4 className="mb-1 font-medium text-amber-600 dark:text-amber-400 text-xs uppercase">
            Sugerencias
          </h4>
          <ul className="flex flex-col gap-1">
            {r.suggestions.map((sug, j) => (
              <li key={j} className="text-amber-700 dark:text-amber-300 text-sm">
                {renderSuggestionWithLinks(sug, knownTopics, j)}
              </li>
            ))}
          </ul>
        </div>
      )}

      {chunks.length > 0 && (
        <div className="mt-3 border-t border-border pt-3">
          <button
            type="button"
            onClick={() => setChunksOpen((o) => !o)}
            className="flex items-center gap-1 text-muted-foreground text-xs hover:text-foreground transition-colors"
          >
            <span>{chunksOpen ? "▲" : "▼"}</span>
            <span>Material de referencia ({chunks.length} fragmento{chunks.length !== 1 ? "s" : ""})</span>
          </button>
          {chunksOpen && (
            <div className="mt-2 flex flex-col gap-2">
              {chunks.map((text, j) => (
                <div key={j} className="rounded-md bg-muted/30 px-3 py-2">
                  <p className="mb-0.5 font-medium text-muted-foreground text-xs uppercase">
                    Fragmento {j + 1}
                  </p>
                  <p className="text-foreground text-xs leading-relaxed whitespace-pre-wrap">{text}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function EvaluationView({
  results,
  knownTopics = [],
}: EvaluationViewProps) {
  const evaluable = results.filter((r) => r.isEvaluable !== false);
  const totalScore = evaluable.reduce((sum, r) => sum + r.score, 0);
  const maxScore = evaluable.length * 10;
  const nonEvaluableCount = results.length - evaluable.length;
  const percentage =
    maxScore > 0 ? Math.round((totalScore / maxScore) * 100) : 0;

  return (
    <div className="flex flex-col gap-6">
      {/* Summary */}
      <div data-testid="total-score" className="rounded-lg border border-border bg-card p-6 text-center">
        <h3 className="font-bold text-2xl text-foreground">
          {totalScore.toFixed(0)} / {maxScore.toFixed(0)}
          <span
            className={cn(
              "ml-2 text-lg",
              percentage >= 70
                ? "text-green-600"
                : percentage >= 50
                  ? "text-amber-600"
                  : "text-red-600",
            )}
          >
            ({percentage}%)
          </span>
        </h3>
        <p className="mt-1 text-muted-foreground text-sm">
          Puntaje total
          {nonEvaluableCount > 0 &&
            ` \u2022 ${nonEvaluableCount} pregunta${nonEvaluableCount !== 1 ? "s" : ""} no evaluable${nonEvaluableCount !== 1 ? "s" : ""}`}
        </p>
      </div>

      {/* Per-question breakdown */}
      {results.map((r, i) => (
        <QuestionCard key={r.questionId} result={r} index={i} knownTopics={knownTopics} />
      ))}

      {results.length === 0 && (
        <p className="text-center text-muted-foreground text-sm">
          No hay resultados para mostrar.
        </p>
      )}
    </div>
  );
}
