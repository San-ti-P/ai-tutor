"use client";

import { useState, useEffect } from "react";
import { BookOpen, TrendingUp, Target, CheckCircle2 } from "lucide-react";
import { useSessionContext } from "@/hooks/SessionProvider";
import { api } from "@/lib/api";
import { StatsCards } from "@/components/dashboard/StatsCards";
import { TopicChart } from "@/components/dashboard/TopicChart";
import { WeakTopics } from "@/components/dashboard/WeakTopics";
import { Spinner } from "@/components/ui/spinner";
import type { StudentProfile } from "@/lib/types";

export default function DashboardPage() {
  const { studentId, activeSession } = useSessionContext();
  const [profile, setProfile] = useState<StudentProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!studentId) return;
    if (!activeSession) {
      setProfile(null);
      setIsLoading(false);
      setError(
        "Todavía no tenés una sesión activa. Creá o seleccioná una sesión para ver tu progreso.",
      );
      return;
    }
    setIsLoading(true);
    api
      .getSessionProfile(activeSession.id)
      .then((res) => {
        // Adapt SessionProfile to match the StudentProfile interface expected by the UI.
        setProfile({
          id: res.data.sessionId,
          topicScores: res.data.topicScores,
          weakTopics: res.data.weakTopics,
          preferences: {
            questionTypes: ["mcq"],
            difficulty: "medium",
            questionCount: 5,
            includeTopics: [],
            excludeTopics: [],
          },
          sessionCount: res.data.examCount,
          sessionHistory: [],
        });
        setError(null);
      })
      .catch(() => {
        setError(
          "Todavía no tenés datos de progreso para esta sesión. Completá un examen para empezar.",
        );
      })
      .finally(() => setIsLoading(false));
  }, [studentId, activeSession?.id]);

  if (isLoading) {
    return (
      <div className="flex flex-col gap-8">
        <div>
          <h1 className="font-bold text-3xl text-foreground tracking-tight">
            Mi Progreso
          </h1>
          <p className="mt-1 text-muted-foreground">
            Seguimiento de tu rendimiento académico
          </p>
        </div>
        <div className="flex flex-col items-center gap-4 py-16">
          <Spinner size="lg" />
          <p className="text-muted-foreground text-sm">Cargando datos...</p>
        </div>
      </div>
    );
  }

  if (error && !profile) {
    return (
      <div className="flex flex-col gap-8">
        <div>
          <h1 className="font-bold text-3xl text-foreground tracking-tight">
            Mi Progreso
          </h1>
          <p className="mt-1 text-muted-foreground">
            Seguimiento de tu rendimiento académico
          </p>
        </div>
        <div className="flex flex-col items-center gap-4 rounded-lg border border-border bg-card p-12">
          <BookOpen className="size-12 text-muted-foreground/40" />
          <p className="text-center text-muted-foreground text-sm">{error}</p>
        </div>
      </div>
    );
  }

  const chartData = profile?.topicScores
    ? (() => {
        // Aggregate leaf scores by root prefix to match weakTopics logic.
        // "cálculo/derivadas": [3, 2] + "cálculo/integrales": [8] →
        // "cálculo": avg(3, 2, 8) = 4.33
        const rootScores: Record<string, number[]> = {};
        for (const [topic, scores] of Object.entries(profile.topicScores)) {
          const root = topic.split("/")[0] ?? topic;
          if (!rootScores[root]) rootScores[root] = [];
          rootScores[root].push(...scores);
        }
        return Object.entries(rootScores)
          .map(([root, scores]) => ({
            topic: root,
            score: scores.reduce((a, b) => a + b, 0) / scores.length,
          }))
          .sort((a, b) => a.score - b.score);
      })()
    : [];

  const topicCount = chartData.length;

  const avgScore =
    chartData.length > 0
      ? chartData.reduce((sum, d) => sum + d.score, 0) / chartData.length
      : null;

  const weakTopics = profile?.weakTopics ?? [];

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="font-bold text-3xl text-foreground tracking-tight">
          Mi Progreso
        </h1>
        <p className="mt-1 text-muted-foreground">
          Seguimiento de tu rendimiento académico
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

      <StatsCards
        stats={[
          {
            label: "Exámenes completados",
            value: String(profile?.sessionCount ?? 0),
            icon: BookOpen,
          },
          {
            label: "Temas cubiertos",
            value: String(topicCount),
            icon: Target,
          },
          {
            label: "Promedio general",
            value: avgScore !== null ? `${(avgScore * 10).toFixed(0)}%` : "--%",
            icon: TrendingUp,
          },
          {
            label: "Ejercicios resueltos",
            value: String(topicCount > 0 ? topicCount : "0"),
            icon: CheckCircle2,
          },
        ]}
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <TopicChart data={chartData} />
        <WeakTopics topics={weakTopics} />
      </div>
    </div>
  );
}
