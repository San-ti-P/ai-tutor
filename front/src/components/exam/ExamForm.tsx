"use client";

import { useState } from "react";
import { Zap } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import type { Difficulty } from "@/lib/types";

const DIFFICULTY_OPTIONS = [
  { value: "easy", label: "F\u00e1cil" },
  { value: "medium", label: "Medio" },
  { value: "hard", label: "Dif\u00edcil" },
];

const QUESTION_TYPES = [
  { value: "mcq", label: "Multiple choice" },
  { value: "open", label: "Respuesta libre" },
];

interface ExamFormProps {
  onSubmit: (data: {
    topic: string;
    difficulty: Difficulty;
    questionCount: number;
    questionTypes: ("mcq" | "open")[];
  }) => void;
  isLoading?: boolean;
  defaultTopic?: string;
}

export function ExamForm({ onSubmit, isLoading, defaultTopic }: ExamFormProps) {
  const [topic, setTopic] = useState(defaultTopic ?? "");
  const [difficulty, setDifficulty] = useState<Difficulty>("medium");
  const [count, setCount] = useState(5);
  const [types, setTypes] = useState<Set<"mcq" | "open">>(new Set(["mcq"]));

  const toggleType = (t: "mcq" | "open") => {
    setTypes((prev) => {
      const next = new Set(prev);
      if (next.has(t)) {
        next.delete(t);
      } else {
        next.add(t);
      }
      // Don't allow empty selection
      return next.size > 0 ? next : prev;
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!topic.trim() || types.size === 0) return;
    onSubmit({
      topic: topic.trim(),
      difficulty,
      questionCount: count,
      questionTypes: Array.from(types),
    });
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-6">
      <div className="flex flex-col gap-1.5">
        <label className="font-medium text-foreground text-sm">
          Tema del examen
        </label>
        <Input
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="Ej: C\u00e1lculo/Derivadas"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <label className="font-medium text-foreground text-sm">
          Dificultad
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
          value={count}
          onChange={(e) => setCount(Math.min(20, Math.max(1, Number(e.target.value) || 1)))}
          className="w-24"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <label className="font-medium text-foreground text-sm">
          Tipo de preguntas
        </label>
        <div className="flex flex-col gap-1">
          {QUESTION_TYPES.map(({ value, label }) => (
            <label
              key={value}
              className="inline-flex items-center gap-2 text-foreground text-sm cursor-pointer"
            >
              <input
                type="checkbox"
                checked={types.has(value as "mcq" | "open")}
                onChange={() => toggleType(value as "mcq" | "open")}
                className="size-3.5 accent-primary"
              />
              {label}
            </label>
          ))}
        </div>
      </div>

      <Button
        type="submit"
        disabled={isLoading || !topic.trim() || types.size === 0}
        className="self-start"
      >
        {isLoading ? (
          <>
            <Spinner size="sm" />
            Generando...
          </>
        ) : (
          <>
            <Zap className="size-4" />
            Generar Examen
          </>
        )}
      </Button>
    </form>
  );
}
