"use client";
import Link from "next/link";
import { Target } from "lucide-react";

interface WeakTopicsProps {
  topics: string[];
}

export function WeakTopics({ topics }: WeakTopicsProps) {
  if (topics.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-6">
        <h3 className="mb-2 font-semibold text-foreground text-sm">
          Temas débiles
        </h3>
        <p className="text-green-600 dark:text-green-400 text-sm">
          ¡Buen trabajo! No tenés temas débiles.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-6">
      <h3 className="mb-3 flex items-center gap-2 font-semibold text-foreground text-sm">
        <Target className="size-4 text-amber-500" />
        Temas para reforzar
      </h3>
      <div className="flex flex-col gap-2">
        {topics.map((topic) => (
          <div
            key={topic}
            className="flex items-center justify-between rounded-md border border-border bg-background px-4 py-2"
          >
            <span className="text-foreground text-sm">{topic}</span>
            <Link
              href={`/exam?topic=${encodeURIComponent(topic)}`}
              className="rounded-md bg-primary px-3 py-1 text-primary-foreground text-xs font-medium hover:bg-primary/90"
            >
              Practicar
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
}
