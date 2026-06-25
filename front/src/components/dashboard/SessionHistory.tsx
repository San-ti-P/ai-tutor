"use client";

import { History } from "lucide-react";

interface SessionEntry {
  id: string;
  started_at: string;
  ended_at: string | null;
  intent: string | null;
  status: string;
  questions_answered: number;
  average_score: number | null;
}

interface SessionHistoryProps {
  sessions: SessionEntry[];
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("es-AR", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function intentLabel(intent: string | null): string {
  switch (intent) {
    case "generate_exam":
      return "Examen";
    case "generate_exercise":
      return "Ejercicio";
    case "evaluate":
      return "Evaluación";
    case "ingest":
      return "Ingesta";
    case "retrieve":
      return "Consulta";
    case "general_chat":
      return "Chat";
    default:
      return intent ?? "Desconocido";
  }
}

export function SessionHistory({ sessions }: SessionHistoryProps) {
  if (sessions.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-6">
        <h3 className="mb-2 flex items-center gap-2 font-semibold text-foreground text-sm">
          <History className="size-4 text-muted-foreground" />
          Historial de sesiones
        </h3>
        <p className="text-muted-foreground text-sm">
          Todavía no tenés sesiones registradas.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-6">
      <h3 className="mb-3 flex items-center gap-2 font-semibold text-foreground text-sm">
        <History className="size-4 text-muted-foreground" />
        Sesiones recientes
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-border text-muted-foreground text-xs">
              <th className="pb-2 pr-3 font-medium">Fecha</th>
              <th className="pb-2 pr-3 font-medium">Tipo</th>
              <th className="pb-2 pr-3 font-medium">Estado</th>
              <th className="pb-2 pr-3 font-medium text-right">
                Preguntas
              </th>
              <th className="pb-2 font-medium text-right">Promedio</th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => (
              <tr
                key={s.id}
                className="border-b border-border/50 last:border-0"
              >
                <td className="py-2 pr-3 text-foreground">
                  {formatDate(s.started_at)}
                </td>
                <td className="py-2 pr-3 text-foreground">
                  {intentLabel(s.intent)}
                </td>
                <td className="py-2 pr-3">
                  <span
                    className={
                      s.status === "completed"
                        ? "text-green-600 dark:text-green-400"
                        : s.status === "active"
                          ? "text-amber-600 dark:text-amber-400"
                          : "text-muted-foreground"
                    }
                  >
                    {s.status === "completed"
                      ? "Completada"
                      : s.status === "active"
                        ? "Activa"
                        : s.status}
                  </span>
                </td>
                <td className="py-2 pr-3 text-right text-foreground">
                  {s.questions_answered}
                </td>
                <td className="py-2 text-right text-foreground">
                  {s.average_score != null
                    ? `${(s.average_score * 10).toFixed(0)}%`
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
