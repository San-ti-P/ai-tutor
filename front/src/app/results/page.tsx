import { BarChart3 } from "lucide-react";

export default function ResultsPage() {
  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="font-bold text-3xl text-foreground tracking-tight">
          Resultados
        </h1>
        <p className="mt-1 text-muted-foreground">
          Historial de exámenes y evaluaciones
        </p>
      </div>

      <div className="flex flex-col items-center gap-4 rounded-lg border border-border bg-card p-12">
        <BarChart3 className="size-12 text-muted-foreground/40" />
        <p className="text-center text-muted-foreground text-sm">
          Completá un examen para ver tus resultados acá.
        </p>
      </div>
    </div>
  );
}
