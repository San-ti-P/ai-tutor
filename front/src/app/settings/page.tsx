"use client";

import { useState, useEffect } from "react";
import { Settings, Save } from "lucide-react";
import { useSessionContext } from "@/hooks/SessionProvider";
import { api } from "@/lib/api";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import type { Difficulty } from "@/lib/types";

const DIFFICULTY_OPTIONS = [
  { value: "easy", label: "Fácil" },
  { value: "medium", label: "Medio" },
  { value: "hard", label: "Difícil" },
];

export default function SettingsPage() {
  const { studentId, activeSession } = useSessionContext();
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [questionTypes, setQuestionTypes] = useState<Set<"mcq" | "open">>(
    new Set(["mcq"]),
  );
  const [difficulty, setDifficulty] = useState<Difficulty>("medium");
  const [questionCount, setQuestionCount] = useState(5);
  const [includeTopics, setIncludeTopics] = useState("");
  const [excludeTopics, setExcludeTopics] = useState("");

  // Load existing preferences on mount
  useEffect(() => {
    if (!studentId) return;
    setIsLoading(true);
    api
      .getProfile(studentId)
      .then((res) => {
        const prefs = res.data.preferences;
        if (prefs) {
          setQuestionTypes(new Set(prefs.questionTypes as ("mcq" | "open")[]));
          setDifficulty(prefs.difficulty);
          setQuestionCount(prefs.questionCount);
          setIncludeTopics(prefs.includeTopics?.join(", ") ?? "");
          setExcludeTopics(prefs.excludeTopics?.join(", ") ?? "");
        }
      })
      .catch(() => {
        // No existing profile — use defaults
      })
      .finally(() => setIsLoading(false));
  }, [studentId]);

  const toggleType = (t: "mcq" | "open") => {
    setQuestionTypes((prev) => {
      const next = new Set(prev);
      if (next.has(t)) {
        next.delete(t);
      } else {
        next.add(t);
      }
      return next.size > 0 ? next : prev;
    });
  };

  const handleSave = async () => {
    if (!studentId) return;
    setIsSaving(true);
    setSaved(false);
    setError(null);

    try {
      await api.updatePreferences(studentId, {
        questionTypes: Array.from(questionTypes) as ("mcq" | "open")[],
        difficulty,
        questionCount,
        includeTopics: includeTopics
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        excludeTopics: excludeTopics
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "No se pudieron guardar las preferencias. Intentá de nuevo.",
      );
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="font-bold text-3xl text-foreground tracking-tight">
          Configuración
        </h1>
        <p className="mt-1 text-muted-foreground">
          Personalizá tu experiencia de estudio
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

      {saved && (
        <div className="rounded-md border border-green-200 bg-green-50 px-4 py-3 text-green-700 text-sm dark:border-green-800 dark:bg-green-950 dark:text-green-300">
          Preferencias guardadas.
        </div>
      )}

      {isLoading ? (
        <div className="flex flex-col items-center gap-4 py-16">
          <Spinner size="lg" />
          <p className="text-muted-foreground text-sm">
            Cargando preferencias...
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-6 rounded-lg border border-border bg-card p-6">
          <div className="flex flex-col gap-1.5">
            <label className="font-medium text-foreground text-sm">
              Tipos de pregunta
            </label>
            <div className="flex flex-col gap-1">
              {[
                { value: "mcq" as const, label: "Multiple choice" },
                { value: "open" as const, label: "Respuesta libre" },
              ].map(({ value, label }) => (
                <label
                  key={value}
                  className="inline-flex items-center gap-2 text-foreground text-sm cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={questionTypes.has(value)}
                    onChange={() => toggleType(value)}
                    className="size-3.5 accent-primary"
                  />
                  {label}
                </label>
              ))}
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="font-medium text-foreground text-sm">
              Dificultad preferida
            </label>
            <Select
              options={DIFFICULTY_OPTIONS}
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value as Difficulty)}
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="font-medium text-foreground text-sm">
              Cantidad de preguntas
            </label>
            <Input
              type="number"
              min={1}
              max={20}
              value={questionCount}
              onChange={(e) =>
                setQuestionCount(
                  Math.min(20, Math.max(1, Number(e.target.value) || 1)),
                )
              }
              className="w-24"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="font-medium text-foreground text-sm">
              Temas a priorizar (separados por coma)
            </label>
            <Input
              type="text"
              value={includeTopics}
              onChange={(e) => setIncludeTopics(e.target.value)}
              placeholder="Álgebra, Cálculo, Probabilidad..."
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="font-medium text-foreground text-sm">
              Temas a excluir (separados por coma)
            </label>
            <Input
              type="text"
              value={excludeTopics}
              onChange={(e) => setExcludeTopics(e.target.value)}
              placeholder="Dejar vacío para no excluir ningún tema"
            />
          </div>

          <Button
            onClick={handleSave}
            disabled={isSaving}
            className="self-start"
          >
            {isSaving ? (
              <>
                <Spinner size="sm" />
                Guardando...
              </>
            ) : (
              <>
                <Save className="size-4" />
                Guardar preferencias
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  );
}
