"use client";

import { useState, useEffect } from "react";
import { BookOpen, TrendingUp, Target, CheckCircle2 } from "lucide-react";
import { useSessionContext } from "@/hooks/SessionProvider";
import { api } from "@/lib/api";
import { StatsCards } from "@/components/dashboard/StatsCards";
import { TopicChart } from "@/components/dashboard/TopicChart";
import { WeakTopics } from "@/components/dashboard/WeakTopics";
import { SessionHistory } from "@/components/dashboard/SessionHistory";
import { Spinner } from "@/components/ui/spinner";
import type { StudentProfile } from "@/lib/types";

export default function DashboardPage() {
  const { studentId, activeSession } = useSessionContext();
  const [profile, setProfile] = useState<StudentProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!studentId) return;
    setIsLoading(true);
    api
      .getDashboard(studentId)
      .then((res) => {
        setProfile(res.data);
      })
      .catch(() => {
        setError(
          "Todavía no tenés datos de progreso. Completá un examen para empezar.",
        );
      })
      .finally(() => setIsLoading(false));
  }, [studentId]);

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

  const topicCount = profile?.topicScores
    ? Object.keys(profile.topicScores).length
    : 0;

  const avgScore =
    profile?.topicScores && topicCount > 0
      ? Object.values(profile.topicScores)
          .flat()
          .reduce((a, b) => a + b, 0) /
        Object.values(profile.topicScores).flat().length
      : null;

  const chartData = profile?.topicScores
    ? Object.entries(profile.topicScores).map(([topic, scores]) => ({
        topic: topic.split("/").pop() ?? topic,
        score: scores.length > 0 ? scores[scores.length - 1] : 0,
      }))
    : [];

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
            label: "Sesiones completadas",
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

      <SessionHistory sessions={profile?.sessionHistory ?? []} />
    </div>
  );
}
